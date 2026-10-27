"""Sanity checks and simple baselines for belief-model training data.

    python3 belief_check.py <dir of .sbd files> [every] [players_per_tick]

Reports how much fog hides, how well cheap guesses recover hidden ships,
and how threat to a player's stars depends on distance to the frontier.
These baselines are the numbers a belief model has to beat.
"""

import sys
from pathlib import Path

import numpy as np

import belief_data as bd

F = {f: i for i, f in enumerate(bd.STAR_FEATURES)}
Y = {f: i for i, f in enumerate(bd.STAR_TARGETS)}


def main():
    src = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    per_tick = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    files = sorted(Path(src).glob("*.sbd"))
    a = bd.build(files, every, per_tick)
    x, y, known = a["star_x"], a["star_y"], a["star_known"]
    ticks = a["meta"][:, 1]
    cycle = bd.load(files[0])[0]["productionTicks"]  # only labels phases
    print(f"{len(files)} games, {len(x)} samples (player views)\n")

    print("How much a player sees, by game phase")
    phases = [(0, 5), (5, 10), (10, 20), (20, 999)]
    enemy = x[..., F["owner_enemy"]] > 0
    for lo, hi in phases:
        m = (ticks >= lo * cycle) & (ticks < hi * cycle)
        if not m.any():
            continue
        vis = x[m][..., F["visible"]].mean()
        seen = x[m][..., F["seen_ever"]].mean()
        ships = y[m][..., Y["ships"]]
        hidden_share = (ships * (enemy[m] & ~known[m])).sum() / max((ships * enemy[m]).sum(), 1e-9)
        print(f"  cycles {lo:>2}-{hi:<3}  in range now {vis:5.1%}   ever seen {seen:5.1%}   "
              f"enemy garrison ships hidden {hidden_share:5.1%}   (n={m.sum()})")

    # Hidden enemy stars: guess their ships.
    print("\nGuessing ships at hidden enemy stars (mean absolute error, in ships)")
    hidden = enemy & ~known
    truth = y[..., Y["ships"]][hidden] * bd.SCALE_SHIPS
    seen = x[..., F["seen_ever"]][hidden] > 0
    last = x[..., F["ships"]][hidden] * bd.SCALE_SHIPS
    print(f"  {hidden.sum()} hidden enemy stars, mean true ships {truth.mean():.1f}")
    print(f"  guess 0                         {np.abs(truth).mean():6.1f}")
    print(f"  guess last seen (0 if never)    {np.abs(truth - last).mean():6.1f}"
          f"   (seen before: {seen.mean():.0%})")
    # Public stats: each enemy's total ships spread evenly over its stars.
    # Includes carriers, so it overestimates garrisons; scale by the share
    # of ships in garrisons across the data.
    even = np.zeros_like(truth)
    k = 0
    px = a["player_x"]
    for i in range(len(x)):
        h = hidden[i]
        if not h.any():
            continue
        n = h.sum()
        # Owner of each hidden star, as a seat in this sample's player order.
        stars_total = px[i, :, 2] * 10
        ships_total = px[i, :, 3] * bd.SCALE_SHIPS
        per_star = np.where(stars_total > 0, ships_total / np.maximum(stars_total, 1), 0)
        even[k:k + n] = per_star[a["star_owner_seat"][i][h]]
        k += n
    garrison_share = truth.sum() / max(even.sum(), 1e-9)
    print(f"  guess owner's public ships / stars  {np.abs(truth - even).mean():6.1f}")
    print(f"  ... scaled by garrison share {garrison_share:.2f}  {np.abs(truth - even * garrison_share).mean():6.1f}")

    # Threat at the player's own stars.
    own = x[..., F["owner_self"]] > 0
    t1 = y[..., Y["threat_1c"]][own] * bd.SCALE_SHIPS
    k1 = x[..., F["known_threat_1c"]][own] * bd.SCALE_SHIPS
    inc = y[..., Y["incoming_1c"]][own] * bd.SCALE_SHIPS
    dist = x[..., F["ticks_to_enemy"]][own]  # in cycles
    print("\nThreat to own stars: enemy ships able to reach within one cycle")
    print(f"  known to the player {k1.mean():.1f} vs true {t1.mean():.1f} on average; "
          f"correlation {np.corrcoef(k1, t1)[0, 1]:.2f}")
    print("\n  By distance to the nearest enemy star (the frontier):")
    print("  cycles away     stars   true threat   known threat   attacked next cycle")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 99)]:
        m = (dist >= lo) & (dist < hi)
        if m.any():
            print(f"  {lo:>4}-{hi:<4}  {m.sum():>12}   {t1[m].mean():>11.1f}   {k1[m].mean():>12.1f}"
                  f"   {(inc[m] > 0).mean():>18.1%}")


if __name__ == "__main__":
    main()
