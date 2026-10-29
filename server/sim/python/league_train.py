"""Train the commander in a league: against frozen copies of itself and of
earlier commanders, as well as the built-in AI.

    python3 league_train.py <run_dir> <start.pt> [iterations] [games] [workers] [eval_games]

start.pt is a trained commander (a ppo_train.py checkpoint). It becomes
the first league member, and the learner starts from it.

Each game seats the learner in LEARNER_SEATS random seats, league members
in OPPONENT_SEATS more (each picked separately), and the built-in AI in the
rest. Members play their most likely decisions under fog, like the learner.
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
import ppo_train as pt

LEARNER_SEATS = 3
OPPONENT_SEATS = 4  # the other 3 seats are the built-in AI
SNAPSHOT_EVERY = 10
RECORD_EVERY = 4
FIRST_SEED = 2_000_000
RATE_DECAY = 0.7  # per iteration: how fast the learner's record against a member follows new games
HARD_POWER = 2.0  # members are picked in proportion to (1 - rate) ** HARD_POWER ...
MIN_PICK = 0.05  # ... plus this, so none is dropped entirely


def load_league(run_dir):
    return json.loads((Path(run_dir) / "league.json").read_text())


def save_league(run_dir, league):
    path = Path(run_dir) / "league.json"
    path.with_suffix(".tmp").write_text(json.dumps(league, indent=1))
    os.replace(path.with_suffix(".tmp"), path)


def add_member(run_dir, league, name, checkpoint):
    members = Path(run_dir) / "members"
    members.mkdir(exist_ok=True)
    path = members / f"{name}.pt"
    saved = torch.load(checkpoint, map_location="cpu")
    torch.save({"model": saved["model"], "n_techs": saved["n_techs"]}, path)
    league["members"].append({"name": name, "path": str(path), "games": 0, "rate": 0.5})


def pick_weights(league):
    rates = np.array([m["rate"] for m in league["members"]])
    w = (1 - rates) ** HARD_POWER + MIN_PICK
    return w / w.sum()


def game_jobs(run_dir, i, games, league, out):
    """Who sits where in each of an iteration's games."""
    rng = np.random.default_rng(i)
    weights = pick_weights(league)
    jobs = []
    for seed in range(FIRST_SEED + i * games, FIRST_SEED + (i + 1) * games):
        seats = rng.permutation(10)
        learner = sorted(int(s) for s in seats[:LEARNER_SEATS])
        members = rng.choice(len(league["members"]), OPPONENT_SEATS, p=weights)
        opponents = {int(s): int(m) for s, m in
                     zip(seats[LEARNER_SEATS:LEARNER_SEATS + OPPONENT_SEATS], members)}
        record = (str(Path(run_dir) / "belief" / f"g{seed}.sbd")
                  if seed % RECORD_EVERY == 0 else None)
        jobs.append((str(out), seed, str(pt.ckpt_path(run_dir, i)), learner, opponents,
                     [m["path"] for m in league["members"]], record))
    return jobs


def _game_job(args):
    out, seed, checkpoint, learner, opponents, paths, record = args
    path = Path(out) / f"r{seed}.npz"
    if not path.exists():
        arrays = cr.play(seed, belief=None, checkpoint=checkpoint, max_ticks=pt.MAX_TICKS,
                         agents=learner, opponents={s: paths[m] for s, m in opponents.items()},
                         record=record)
        np.savez_compressed(str(path) + ".tmp.npz", **arrays)
        os.replace(str(path) + ".tmp.npz", path)
    return seed, opponents


def collect(run_dir, i, games, workers, league):
    out = Path(run_dir) / f"iter_{i}"
    out.mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "belief").mkdir(exist_ok=True)
    jobs = game_jobs(run_dir, i, games, league, out)
    seating = {}
    with Pool(workers) as pool:
        for n, (seed, opponents) in enumerate(pool.imap_unordered(_game_job, jobs), 1):
            seating[seed] = opponents
            print(f"  played {n}/{games}", end="\r", flush=True)
    print()
    return [(out / f"r{seed}.npz", seating[seed]) for seed in sorted(seating)]


def score(league, games):
    """Update the learner's record against each member it met: for each
    pairing of a learner seat with a member seat, 1 if the learner placed
    above, 0 if below."""
    results = {}
    for path, opponents in games:
        z = np.load(path)
        place = z["place_all"]
        for s in z["agents"]:
            for o, m in opponents.items():
                results.setdefault(m, []).append(float(place[s] < place[o]))
    for m, r in results.items():
        member = league["members"][m]
        member["rate"] = RATE_DECAY * member["rate"] + (1 - RATE_DECAY) * float(np.mean(r))
        member["games"] += len(r)
    return {league["members"][m]["name"]: round(float(np.mean(r)), 3) for m, r in results.items()}


def main():
    run_dir = sys.argv[1]
    start_ckpt = sys.argv[2]
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 52
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else max(1, os.cpu_count() - 1)
    eval_games = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(run_dir, exist_ok=True)

    if not (Path(run_dir) / "league.json").exists():
        league = {"members": []}
        add_member(run_dir, league, "start", start_ckpt)
        shutil.copy(league["members"][0]["path"], pt.ckpt_path(run_dir, 0))
        save_league(run_dir, league)
    league = load_league(run_dir)

    start = pt.latest(run_dir)
    saved = torch.load(pt.ckpt_path(run_dir, start), map_location="cpu")
    techs = saved["n_techs"]
    model = cr.build_model(techs)
    model.load_state_dict(saved["model"])
    model.to(device)
    opt = pt.optimizer(model)
    if "optimizer" in saved:
        opt.load_state_dict(saved["optimizer"])
    print(f"league run {run_dir}: from iteration {start}, {len(league['members'])} members, "
          f"on {device}", flush=True)

    def log(name, record):
        with open(Path(run_dir) / name, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in record.items()), flush=True)

    for i in range(start, start + iterations):
        if eval_games and i % pt.EVAL_EVERY == 0:
            print(f"evaluating ckpt_{i} against the built-in AI", flush=True)
            log("eval.jsonl", pt.evaluate(run_dir, i, eval_games, workers))

        t0 = time.time()
        print(f"iteration {i}: playing {games} league games", flush=True)
        played = collect(run_dir, i, games, workers, league)
        against = score(league, played)
        steps = pt.Steps([p for p, _ in played])
        t1 = time.time()
        model.eval()  # no dropout: the recorded log-probabilities must match
        stats = pt.update(model, opt, steps, device)
        torch.save({"model": model.state_dict(), "n_techs": techs,
                    "optimizer": opt.state_dict()}, pt.ckpt_path(run_dir, i + 1))
        if (i + 1) % SNAPSHOT_EVERY == 0:
            add_member(run_dir, league, f"ckpt_{i + 1}", pt.ckpt_path(run_dir, i + 1))
        save_league(run_dir, league)
        log("log.jsonl", {"iteration": i, "placement": float(np.mean(steps.places)),
                          "placed_above": against, **stats,
                          "play_s": round(t1 - t0), "learn_s": round(time.time() - t1)})

    if eval_games:
        print(f"evaluating ckpt_{start + iterations} against the built-in AI", flush=True)
        log("eval.jsonl", pt.evaluate(run_dir, start + iterations, eval_games, workers))


if __name__ == "__main__":
    main()
