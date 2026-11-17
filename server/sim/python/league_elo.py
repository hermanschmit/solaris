"""Elo ratings for a league run, from the games it has already played.

    python3 league_elo.py <run_dir> [block]

Every league game seats the learner, league members and the built-in AI
together, and records every seat's placement. Each pair of seats held by
different players is one result: the one placed higher beat the other.
The ratings are the Bradley-Terry fit to all those results (Elo's model:
a 400-point gap means 10 to 1 odds), with the built-in AI pinned at 1000.

The learner changes every iteration, so it is rated in blocks of `block`
iterations (default 10, the same as SNAPSHOT_EVERY): "learner 100-109" is
the checkpoints that played iterations 100 to 109. Members are frozen, so
each is one player throughout. Built-in AI seats playing a flawed persona
(personas.py) are rated per persona.

Nothing here changes training; run it while a league trains. Results
within one game are not independent, so the ± (a bootstrap over games)
is the one to trust, not the game counts.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

BUILTIN = "built-in AI"
ANCHOR = 1000.0
SCALE = 400.0 / np.log(10)  # Elo points per natural-log unit of odds
PRIOR_SD = 600.0  # a weak pull toward the built-in AI, for players with few games
BOOTSTRAP = 50


def results(run_dir, block):
    """Per game, the list of (winner, loser) player names."""
    league = json.loads((Path(run_dir) / "league.json").read_text())
    names = [m["name"] for m in league["members"]]
    games = []
    for d in sorted(Path(run_dir).glob("iter_*")):
        i = int(re.sub(r"\D", "", d.name))
        start = i - i % block
        learner = f"learner {start}-{start + block - 1}"
        for path in d.glob("r*.npz"):
            try:
                z = np.load(path)
                if "opponent_seats" not in z.files or "place_all" not in z.files:
                    continue
                place = z["place_all"]
                who = [BUILTIN] * len(place)
                for s in z["agents"]:
                    who[int(s)] = learner
                for s, m in zip(z["opponent_seats"], z["opponent_members"]):
                    who[int(s)] = names[int(m)]
                if "persona_seats" in z.files:
                    for s, name in zip(z["persona_seats"], z["persona_names"]):
                        who[int(s)] = f"persona {name}"
            except (OSError, ValueError, KeyError, IndexError):
                continue  # a game still being written
            pairs = [(who[a], who[b]) if place[a] < place[b] else (who[b], who[a])
                     for a in range(len(place)) for b in range(a + 1, len(place))
                     if who[a] != who[b] and place[a] != place[b]]
            games.append(pairs)
    return games


def fit(games, players):
    """Bradley-Terry ratings (Elo points), built-in AI pinned at ANCHOR."""
    index = {p: k for k, p in enumerate(players)}
    win = np.array([index[w] for g in games for w, _ in g], int)
    lose = np.array([index[l] for g in games for _, l in g], int)
    n = len(players)
    r = np.zeros(n)  # natural-log units, relative to the built-in AI
    fixed = index[BUILTIN]
    prior = (PRIOR_SD / SCALE) ** -2
    for _ in range(200):  # Newton steps on the penalized log-likelihood
        p = 1 / (1 + np.exp(r[lose] - r[win]))  # P(winner beats loser)
        g = np.bincount(win, 1 - p, n) - np.bincount(lose, 1 - p, n) - prior * r
        w = p * (1 - p)
        h = np.bincount(win, w, n) + np.bincount(lose, w, n) + prior
        step = g / h
        step[fixed] = 0
        r += step
        if np.abs(step).max() < 1e-6:
            break
    return ANCHOR + SCALE * r


def main():
    run_dir = sys.argv[1]
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    games = results(run_dir, block)
    if not games:
        sys.exit("No scored league games found.")
    players = sorted({p for g in games for pair in g for p in pair})
    rating = fit(games, players)

    rng = np.random.default_rng(0)
    boot = np.array([fit([games[k] for k in rng.integers(0, len(games), len(games))], players)
                     for _ in range(BOOTSTRAP)])
    sd = boot.std(0)
    count = {p: 0 for p in players}
    for g in games:
        for p in {x for pair in g for x in pair}:
            count[p] += 1

    def order(p):
        m = re.search(r"(\d+)", p)
        return (p == BUILTIN, p.startswith("persona"), p.startswith("learner"),
                int(m.group(1)) if m else -1, p)

    print(f"{len(games)} games\n")
    print(f"{'player':<22}{'rating':>8}{'±':>6}{'games':>7}")
    for p in sorted(players, key=order):
        k = players.index(p)
        print(f"{p:<22}{rating[k]:>8.0f}{sd[k]:>6.0f}{count[p]:>7}")
    out = Path(run_dir) / "elo.json"
    out.write_text(json.dumps({p: {"rating": round(float(rating[k]), 1), "sd": round(float(sd[k]), 1),
                                   "games": count[p]} for k, p in enumerate(players)}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
