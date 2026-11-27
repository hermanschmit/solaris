"""Train the commander in a league: against frozen copies of itself and of
earlier commanders, as well as the built-in AI.

    python3 league_train.py <run_dir> <start.pt> [iterations] [games] [workers] [eval_games]

start.pt is a trained commander (a ppo_train.py checkpoint). It becomes
the first league member, and the learner starts from it.

Each game seats the learner in LEARNER_SEATS random seats, league members
in OPPONENT_SEATS more (each picked separately), and the built-in AI in the
rest; in PERSONA_SHARE of games, some of those play a flawed persona
(personas.py). Members play their most likely decisions under fog, like the learner.
A preset with a league_mix (1v1) instead picks each game's opponents as a
whole: league members, the built-in AI, or the built-in AI with a flawed
persona, in the mix's shares.
Members the learner does badly against are picked more often, so it keeps
working on what it cannot yet beat. Every SNAPSHOT_EVERY iterations the
learner joins the league as a new member.

Learning is the same PPO as ppo_train.py. Files in run_dir:

  league.json             the members, and how the learner does against
                          each (the share of games it placed above them)
  ckpt_<i>.pt             the learner after i iterations
  iter_<i>/               the games played in iteration i
  belief/g<seed>.sbd      one game in RECORD_EVERY, recorded for retraining
                          the belief model on league play (belief_data.py
                          reads this directory like beliefBatch.ts output)
  log.jsonl, eval.jsonl   as in ppo_train.py; evaluation is still against
                          the built-in AI seeing the whole game

Run it again with the same run_dir to continue.
"""

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import commander_rollout as cr
import personas as pa
import presets
import ppo_train as pt

LEARNER_SEATS = presets.SEATS
OPPONENT_SEATS = presets.PRESET["league_opponents"]  # the rest are the built-in AI
SNAPSHOT_EVERY = 10
RECORD_EVERY = 4
FIRST_SEED = 2_000_000
RATE_DECAY = 0.7  # per iteration: how fast the learner's record against a member follows new games
HARD_POWER = 2.0  # members are picked in proportion to (1 - rate) ** HARD_POWER ...
MIN_PICK = 0.05  # ... plus this, so none is dropped entirely
# In this share of games, 1 or 2 of the built-in AI seats play a flawed
# persona (personas.TRAIN), so the learner practises punishing mistakes;
# the other games are as before, so it keeps practising against sound play.
PERSONA_SHARE = 0.5


def load_league(run_dir):
    return json.loads((Path(run_dir) / "league.json").read_text())


def save_league(run_dir, league):
    path = Path(run_dir) / "league.json"
    path.with_suffix(".tmp").write_text(json.dumps(league, indent=1))
    os.replace(path.with_suffix(".tmp"), path)


def add_member(run_dir, league, name, checkpoint, kind=None):
    members = Path(run_dir) / "members"
    members.mkdir(exist_ok=True)
    path = members / f"{name}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    detected_kind = kind or (
        "admiral"
        if isinstance(saved, dict) and str(saved.get("arch", "")).startswith("admiral")
        else "commander"
    )
    if detected_kind == "admiral":
        payload = {
            "arch": saved.get("arch", "admiral"),
            "config": saved.get("config", {}),
            "model": saved["model"] if "model" in saved else saved,
            "iteration": saved.get("iteration", 0),
        }
        torch.save(payload, path)
    else:
        torch.save({"model": saved["model"], "n_techs": saved["n_techs"]}, path)
    league["members"].append({
        "name": name,
        "path": str(path),
        "kind": detected_kind,
        "games": 0,
        "rate": 0.5,
    })


def pick_weights(league, indices=None):
    members = league["members"]
    if indices is None:
        indices = list(range(len(members)))
    rates = np.array([members[idx]["rate"] for idx in indices], dtype=np.float64)
    w = (1 - rates) ** HARD_POWER + MIN_PICK
    return w / w.sum()


def game_maker(run_dir, league, max_admiral_seats=1, admiral_share=None):
    """Who sits where in the k-th game started in iteration i, by the
    league as it is when the game starts."""
    def make_job(i, k):
        seed = pt.train_seed(FIRST_SEED, i, k)
        rng = np.random.default_rng(seed)
        seats = rng.permutation(presets.PLAYERS)
        learner = sorted(int(s) for s in seats[:LEARNER_SEATS])

        admiral_indices = [
            idx for idx, m in enumerate(league["members"])
            if m.get("kind") == "admiral"
        ]
        commander_indices = [
            idx for idx, m in enumerate(league["members"])
            if m.get("kind", "commander") != "admiral"
        ]

        chosen_members = []
        opp_seat_list = list(seats[LEARNER_SEATS:LEARNER_SEATS + OPPONENT_SEATS])
        adm_cap = min(max_admiral_seats, OPPONENT_SEATS) if admiral_indices else 0

        if adm_cap > 0 and admiral_share is not None:
            # Explicit curriculum share: include up to `adm_cap` Admiral seats in
            # `admiral_share` fraction of games, filling remaining opponent slots
            # with Commander league members.
            n_adm = adm_cap if rng.random() < admiral_share else 0
            if n_adm > 0:
                adm_pick = rng.choice(
                    admiral_indices, n_adm,
                    p=pick_weights(league, admiral_indices),
                )
                chosen_members.extend(int(m) for m in adm_pick)
            rem = OPPONENT_SEATS - len(chosen_members)
            if rem > 0:
                pool = commander_indices or admiral_indices
                cmd_pick = rng.choice(pool, rem, p=pick_weights(league, pool))
                chosen_members.extend(int(m) for m in cmd_pick)
        else:
            # Adaptive league weights, capping Admiral seats per game at `adm_cap`
            # so a weaker Commander is not overwhelmed by multiple Admirals at once.
            adm_count = 0
            for _ in range(OPPONENT_SEATS):
                if adm_count >= adm_cap and commander_indices:
                    m = int(rng.choice(
                        commander_indices, p=pick_weights(league, commander_indices),
                    ))
                else:
                    m = int(rng.choice(len(league["members"]), p=pick_weights(league)))
                if league["members"][m].get("kind") == "admiral":
                    adm_count += 1
                chosen_members.append(m)

        rng.shuffle(chosen_members)
        opponents = {int(s): int(m) for s, m in zip(opp_seat_list, chosen_members)}
        builtin = seats[LEARNER_SEATS + OPPONENT_SEATS:]
        flawed = {}
        mix = presets.PRESET.get("league_mix")
        if mix:
            kind = rng.choice(sorted(mix), p=[mix[k] for k in sorted(mix)])
            if kind != "member":
                builtin = seats[LEARNER_SEATS:]
                opponents = {}
            if kind == "persona":
                flawed = pa.assign(rng, builtin)
        elif rng.random() < PERSONA_SHARE:
            flawed = pa.assign(rng, builtin[:int(rng.integers(1, 3))])
        out = Path(run_dir) / f"iter_{i}"
        out.mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "belief").mkdir(exist_ok=True)
        record = (str(Path(run_dir) / "belief" / f"g{seed}.sbd")
                  if seed % RECORD_EVERY == 0 else None)
        member_specs = [
            (m["path"], m.get("kind", "commander")) for m in league["members"]
        ]
        return (str(out), seed, str(pt.ckpt_path(run_dir, i)), learner, opponents,
                member_specs, record, flawed)
    return make_job


def _game_job(args):
    out, seed, checkpoint, learner, opponents, member_specs, record, flawed = args
    path = Path(out) / f"r{seed}.npz"
    if not path.exists():
        cmd_opponents = {
            s: member_specs[m][0]
            for s, m in opponents.items()
            if member_specs[m][1] != "admiral"
        }
        adm_opponents = {
            s: member_specs[m][0]
            for s, m in opponents.items()
            if member_specs[m][1] == "admiral"
        }
        arrays = cr.play(seed, belief=None, checkpoint=checkpoint, max_ticks=pt.MAX_TICKS,
                         agents=learner, opponents=cmd_opponents,
                         admiral_opponents=adm_opponents,
                         record=record,
                         ai_configs={s: config for s, (_, config) in flawed.items()})
        # Which league member sat where, for scoring the learner against it.
        arrays["opponent_seats"] = np.array(list(opponents), int)
        arrays["opponent_members"] = np.array(list(opponents.values()), int)
        arrays["admiral_seats"] = np.array(list(adm_opponents), int)
        # Which built-in seats played a flawed persona, for league_elo.py.
        arrays["persona_seats"] = np.array(list(flawed), int)
        arrays["persona_names"] = np.array([name for name, _ in flawed.values()], dtype="U16")
        np.savez_compressed(str(path) + ".tmp.npz", **arrays)
        os.replace(str(path) + ".tmp.npz", path)
    return path


def score(league, games):
    """Update the learner's record against each member it met: for each
    pairing of a learner seat with a member seat, 1 if the learner placed
    above, 0 if below."""
    results = {}
    for path in games:
        z = np.load(path)
        if "opponent_seats" not in z.files:  # recorded before seating was kept
            continue
        place = z["place_all"]
        for s in z["agents"]:
            for o, m in zip(z["opponent_seats"], z["opponent_members"]):
                results.setdefault(int(m), []).append(float(place[s] < place[o]))
    for m, r in results.items():
        member = league["members"][m]
        member["rate"] = RATE_DECAY * member["rate"] + (1 - RATE_DECAY) * float(np.mean(r))
        member["games"] += len(r)
    return {league["members"][m]["name"]: round(float(np.mean(r)), 3) for m, r in results.items()}


def placement_by_opponent(games):
    """The learner's mean placement in games against Admiral opponents,
    Commander league members, the built-in AI and flawed personas."""
    out = {}
    for path in games:
        z = np.load(path)
        kind = ("admiral" if ("admiral_seats" in z.files and len(z["admiral_seats"])) else
                "member" if len(z.get("opponent_seats", ())) else
                "persona" if len(z.get("persona_seats", ())) else "builtin")
        out.setdefault(kind, []).extend(z["place"].tolist())
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(out.items())}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Train Commander in a league (frozen Commanders, built-in AI/personas, and optional AdmiralNet opponents)."
    )
    parser.add_argument("run_dir", help="Output directory for checkpoints, league.json, and logs")
    parser.add_argument("start_ckpt", help="Starting Commander checkpoint (.pt)")
    parser.add_argument("iterations", nargs="?", type=int, default=50, help="Number of PPO iterations (default: 50)")
    parser.add_argument("games", nargs="?", type=int, default=52, help="League games per iteration (default: 52)")
    parser.add_argument("workers", nargs="?", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel worker processes (default: cpu_count - 1)")
    parser.add_argument("eval_games", nargs="?", type=int, default=100,
                        help="Evaluation games against built-in AI (default: 100)")
    parser.add_argument("--admiral", action="append", default=[], metavar="CKPT",
                        help="Path to a frozen AdmiralNet checkpoint (.pt) to add as a league opponent (repeatable)")
    parser.add_argument("--admiral-seats", type=int, default=1,
                        help="Max Admiral opponent seats per game (default: 1)")
    parser.add_argument("--admiral-share", type=float, default=None,
                        help="Optional fraction of games [0..1] guaranteed to include --admiral-seats Admiral opponent(s)")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    run_dir = args.run_dir
    start_ckpt = args.start_ckpt
    iterations = args.iterations
    games = args.games
    workers = args.workers
    eval_games = args.eval_games
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(run_dir, exist_ok=True)

    if not (Path(run_dir) / "league.json").exists():
        league = {"members": []}
        add_member(run_dir, league, "start", start_ckpt, kind="commander")
        shutil.copy(league["members"][0]["path"], pt.ckpt_path(run_dir, 0))
        save_league(run_dir, league)
    league = load_league(run_dir)

    # Register any --admiral checkpoints not already present in the league.
    existing_names = {m["name"] for m in league["members"]}
    added_admiral = False
    for adm_path in args.admiral:
        stem = Path(adm_path).stem
        parent = Path(adm_path).parent.name
        name = f"admiral_{parent}_{stem}" if parent else f"admiral_{stem}"
        if name not in existing_names:
            add_member(run_dir, league, name, adm_path, kind="admiral")
            existing_names.add(name)
            added_admiral = True
    if added_admiral:
        save_league(run_dir, league)

    start = pt.latest(run_dir)
    saved = torch.load(pt.ckpt_path(run_dir, start), map_location="cpu", weights_only=False)
    techs = saved["n_techs"]
    model = cr.build_model(techs)
    cr.load_commander(model, saved["model"])
    model.to(device)
    opt = pt.optimizer(model)
    ema = pt.ema_start(model, saved)
    if "optimizer" in saved:
        try:
            opt.load_state_dict(saved["optimizer"])
        except ValueError:
            print("The checkpoint has fewer heads than the commander now: new optimizer state.")
    n_adm = sum(1 for m in league["members"] if m.get("kind") == "admiral")
    print(f"league run {run_dir}: from iteration {start}, {len(league['members'])} members "
          f"({n_adm} admiral, max_admiral_seats={args.admiral_seats}), on {device}", flush=True)

    def log(name, record):
        with open(Path(run_dir) / name, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in record.items()), flush=True)

    with Pool(workers) as pool:
        collector = pt.Collector(
            pool, workers, _game_job,
            game_maker(run_dir, league, max_admiral_seats=args.admiral_seats,
                       admiral_share=args.admiral_share),
        )
        for i in range(start, start + iterations):
            pt.evaluations(run_dir, i, eval_games, workers, pool, log)

            t0 = time.time()
            print(f"iteration {i}: playing {games} league games", flush=True)
            collector.start(i)
            played = collector.take(games)
            paths = [path for _, path in played]
            with collector.lock:  # games starting meanwhile read the league
                against = score(league, paths)
            steps = pt.Steps(paths)
            t1 = time.time()
            model.eval()  # no dropout: the recorded log-probabilities must match
            stats = pt.update(model, opt, steps, device)
            pt.ema_update(ema, model)
            pt.save_checkpoint(run_dir, i + 1, model, opt, ema, techs)
            with collector.lock:
                if (i + 1) % SNAPSHOT_EVERY == 0:
                    # The averaged weights, as the evaluations play: the
                    # trained weights just after one update can be a dip.
                    add_member(run_dir, league, f"ema_{i + 1}", pt.ema_path(run_dir, i + 1),
                               kind="commander")
                save_league(run_dir, league)
            log("log.jsonl", {"iteration": i, "placement": float(np.mean(steps.places)),
                              "hired": steps.hired, **pt.strategy_shares(steps),
                              "placed_above": against,
                              "placement_by_opponent": placement_by_opponent(paths),
                              "lag": float(np.mean([i - j for j, _ in played])), **stats,
                              "play_s": round(t1 - t0), "learn_s": round(time.time() - t1)})

        if eval_games and (start + iterations) % pt.EVAL_EVERY:
            print(f"evaluating ckpt_{start + iterations} against the built-in AI", flush=True)
            log("eval.jsonl", pt.evaluate(run_dir, start + iterations, eval_games, workers, pool))


if __name__ == "__main__":
    main()
