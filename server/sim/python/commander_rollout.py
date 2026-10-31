"""Play games with the commander and record them for training.

    python3 commander_rollout.py <out_dir> [games] [first_seed] [workers] [seats] [belief.pt] [commander.pt]

Each game puts the commander in `seats` random seats (3 by default) under
fog of war, against the built-in AI in the rest, and samples its decisions
so training can learn from what it tried. Per decision it records the
policy's inputs (the seat's view only), what it chose and how likely that
was, and the critic's inputs (the true state). At the end it records each
seat's placement and the reward:

  reward = 1 for a win down to -1 for first out, by placement, plus a small
  bonus each cycle for the share of stars the seat holds.

Games run in parallel, one per worker process, and each writes
<out_dir>/r<seed>.npz. A commander checkpoint, when given, is the policy
being trained; otherwise it starts from scratch (which plays exactly like
the built-in AI, give or take the sampling). The belief model, when given,
initializes its star encoding.
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import belief_data as bd
import commander as cm
from solaris_env import SolarisEnv

STAR_SHARE_BONUS = 0.02  # per cycle, times the seat's share of all stars


def placement_reward(place, players):
    return 1.0 - 2.0 * (place - 1) / (players - 1)


def placements(standings):
    key = lambda p: (standings["winner"] != p["seat"], p["defeated"], -p["stars"])
    order = sorted(standings["players"], key=key)
    return {p["seat"]: i + 1 for i, p in enumerate(order)}


def build_model(n_techs, belief=None, checkpoint=None):
    model = cm.Commander(n_techs)
    if belief:
        model.load_belief(belief)
    if checkpoint:
        load_commander(model, torch.load(checkpoint, map_location="cpu")["model"])
    return model


def load_commander(model, state):
    """Load a commander's weights. Heads added since the checkpoint was
    saved keep their starting weights, which leave the built-in AI's
    choices alone."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    new_heads = ("plan_head.", "specialist_head.", "specialist_star_head.")
    if unexpected or any(not k.startswith(new_heads) for k in missing):
        raise ValueError(f"Checkpoint does not fit: missing {missing}, unexpected {unexpected}")


def play(seed, seats=3, belief=None, checkpoint=None, max_ticks=3000, agents=None,
         opponents=None, record=None):
    """Play a game with the commander being trained in `agents` (or
    `seats` random seats), sampling its decisions, and record them.
    `opponents` maps other seats to frozen commander checkpoints, which
    play their most likely decisions under fog; the built-in AI plays the
    rest. With `record`, the game is also saved there for belief-model
    training."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    opponents = opponents or {}
    if agents is None:
        agents = sorted(int(s) for s in rng.choice(10, seats, replace=False))
    with SolarisEnv() as env:
        result = env.reset(agents=sorted(list(agents) + list(opponents)), seed=seed,
                           max_ticks=max_ticks, fog=True, record=record)
        static = bd.Static(env.header())
        model = build_model(len(static.techs), belief, checkpoint)
        agent = cm.CommanderAgent(model, static, sample=True)
        frozen = {path: cm.CommanderAgent(build_model(len(static.techs), None, path), static)
                  for path in set(opponents.values())}
        torch.manual_seed(seed)

        steps = []
        last_cycle = {}
        while not result["done"]:
            for seat in result["seats"]:
                if seat in opponents:
                    obs = env.observe(seat, frame=True)
                    env.act(seat, frozen[opponents[seat]].decide(obs, seat)[0])
                    continue

                obs = env.observe(seat, frame=True, truth_frame=True)
                decision, rec = agent.decide(obs, seat)
                env.act(seat, decision)

                truth = cm.batch_inputs(static, obs["truthFrame"], seat, obs["candidates"],
                                        obs["tick"], False)
                bonus = 0.0
                if obs["cycle"] != last_cycle.get(seat):
                    last_cycle[seat] = obs["cycle"]
                    own = sum(1 for o in obs["frame"]["owner"] if o == seat)
                    bonus = STAR_SHARE_BONUS * own / static.S
                steps.append((seat, obs["tick"], rec, truth, bonus))
            result = env.step()

        standings = env.standings()

    place = placements(standings)
    return pack(steps, place, agents, seed, standings)


def pack(steps, place, agents, seed, standings):
    """Stack a game's decisions into arrays, padding candidate lists."""
    C = max(r["inputs"]["cand_mask"].shape[1] for _, _, r, _, _ in steps)

    def pad(x, value=0):
        x = x[0].numpy()
        if x.ndim and x.shape[0] < C and x.shape[0] != 0:
            x = np.concatenate([x, np.full((C - x.shape[0],) + x.shape[1:], value, x.dtype)])
        return x

    out = {k: [] for k in ("seat", "tick", "logp", "bonus")}
    inputs, truths = {}, {}
    for seat, tick, rec, truth, bonus in steps:
        out["seat"].append(seat)
        out["tick"].append(tick)
        out["logp"].append(float(rec["logp"][0]))
        out["bonus"].append(bonus)
        for k, v in rec["act"].items():
            out.setdefault("act_" + k, []).append(pad(v) if k == "order" else v[0].numpy())
        for k, v in rec["inputs"].items():
            inputs.setdefault(k, []).append(pad(v, -1 if k == "cand_star" else 0) if k.startswith("cand_") else v[0].numpy())
        for k in ("star_x", "player_x", "owner", "time"):
            truths.setdefault(k, []).append(truth[k][0].numpy())

    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays.update({"in_" + k: np.stack(v) for k, v in inputs.items()})
    arrays.update({"truth_" + k: np.stack(v) for k, v in truths.items()})
    players = len(standings["players"])
    arrays["agents"] = np.array(agents)
    arrays["place"] = np.array([place[s] for s in agents])
    arrays["final_reward"] = np.array([placement_reward(place[s], players) for s in agents],
                                      np.float32)
    arrays["hired"] = np.array([standings["players"][s].get("hired", 0) for s in agents])
    arrays["hires_refused"] = np.array([standings["players"][s].get("hiresRefused", 0)
                                        for s in agents])
    arrays["place_all"] = np.array([place[p["seat"]] for p in standings["players"]])
    arrays["seed"] = np.array(seed)
    arrays["ticks"] = np.array(standings["tick"])
    return arrays


def eval_seats(seed, seats=3):
    """The seats an evaluation game on this seed gives the commander."""
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(10, seats, replace=False))


def play_eval(seed, seats=3, checkpoint=None, max_ticks=3000, sample=False):
    """Play a seed's evaluation seats with the commander's most likely
    decisions (or as `sample` says, see CommanderAgent), under fog, and
    return their placements. Without a checkpoint, the built-in AI plays
    every seat with the whole game in view: the bar to beat."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    agents = eval_seats(seed, seats)
    with SolarisEnv() as env:
        result = env.reset(agents=agents if checkpoint else [], seed=seed,
                           max_ticks=max_ticks, fog=True)
        if checkpoint:
            static = bd.Static(env.header())
            agent = cm.CommanderAgent(build_model(len(static.techs), None, checkpoint), static,
                                      sample=sample)
        while not result["done"]:
            for seat in result["seats"]:
                obs = env.observe(seat, frame=True)
                env.act(seat, agent.decide(obs, seat)[0])
            result = env.step()
        place = placements(env.standings())
    return {s: place[s] for s in agents}


def _job(args):
    out_dir, seed, seats, belief, checkpoint, max_ticks = args
    path = Path(out_dir) / f"r{seed}.npz"
    if path.exists():
        return seed, None
    arrays = play(seed, seats, belief, checkpoint, max_ticks)
    np.savez_compressed(str(path) + ".tmp.npz", **arrays)
    os.replace(str(path) + ".tmp.npz", path)
    return seed, dict(zip(arrays["agents"].tolist(), arrays["place"].tolist()))


if __name__ == "__main__":
    out_dir = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, os.cpu_count() - 1)
    seats = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    belief = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    checkpoint = sys.argv[7] if len(sys.argv) > 7 else None
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(out_dir, s, seats, belief, checkpoint, 3000) for s in range(first, first + games)]
    with Pool(workers) as pool:
        for seed, places in pool.imap_unordered(_job, jobs):
            print(f"seed {seed}: " + ("already done" if places is None else
                  "placements " + ", ".join(f"seat {s} -> {p}" for s, p in places.items())),
                  flush=True)
