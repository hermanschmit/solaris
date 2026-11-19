"""Play two commander checkpoints against each other.

    SOLARIS_SETTINGS=1v1 python3 head_to_head.py <a.pt> <b.pt> [seeds] [workers] [first_seed]

Each seed is played twice, the checkpoints swapping roles: in 1v1 that
swaps seats, so the map's and the seat's advantages cancel. Both play their most likely
decisions under fog, as in evaluation. In games with more than two
players, the other seats are the built-in AI and each checkpoint plays
presets.SEATS seats. Prints the share of (seat, rival seat) pairs where a
placed above b, with its standard error over seeds, and each side's mean
placement. Seeds start at 800,000, away from training and evaluation.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

import commander_rollout as cr
import presets

MAX_TICKS = int(os.environ.get("PPO_MAX_TICKS", 3000))


def _job(args):
    seed, ours, theirs = args
    return seed, ours, cr.play_eval(seed, presets.SEATS, ours, MAX_TICKS, opponent=theirs)


def score(places):
    """a's share of pairs placed above b, and the mean placements, from
    one game's play_eval result with a as the agent."""
    mine = [p for k, p in places.items() if k >= 0]
    theirs = [p for k, p in places.items() if k < 0]
    pairs = [1.0 if x < y else 0.5 if x == y else 0.0 for x in mine for y in theirs]
    return float(np.mean(pairs)), float(np.mean(mine)), float(np.mean(theirs))


def main():
    a, b = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, os.cpu_count() - 1)
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 800_000
    jobs = [(s, x, y) for s in range(first, first + seeds) for x, y in ((a, b), (b, a))]
    per_seed = {}
    with Pool(workers) as pool:
        for n, (seed, ours, places) in enumerate(pool.imap_unordered(_job, jobs), 1):
            above, mine, theirs = score(places)
            if ours != a:  # b was the agent: turn it round
                above, mine, theirs = 1 - above, theirs, mine
            per_seed.setdefault(seed, []).append((above, mine, theirs))
            print(f"\r  played {n}/{len(jobs)}", end="", flush=True)
    print()
    rows = np.array([np.mean(v, 0) for v in per_seed.values()])
    above = rows[:, 0]
    print(f"{presets.NAME}: {a} placed above {b} in {above.mean():.3f} "
          f"± {above.std() / len(above) ** 0.5:.3f} of pairs over {len(above)} seeds "
          f"(both sides each); mean placement {rows[:, 1].mean():.3f} vs {rows[:, 2].mean():.3f}")


if __name__ == "__main__":
    main()
