"""A toy agent: seat 0 plays the built-in AI's orders but prefers invasions
and researches weapons, against nine default built-in AIs.

    python3 sim/python/example_agent.py [seed] [max_ticks]
"""

import sys
import time

from solaris_env import SolarisEnv


def decide(obs):
    orders = {}
    for c in obs["candidates"]:
        bonus = 2.0 if c["kind"] == "invade" else 1.0
        orders[c["key"]] = max(orders.get(c["key"], 0), c["score"] * bonus)
    return {
        "orders": orders,
        "config": {"invasionAttackFactor": 1.3},
        "research": "weapons",
    }


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    max_ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    start = time.time()

    with SolarisEnv() as env:
        result = env.reset(agents=[0], seed=seed, max_ticks=max_ticks)
        decisions = 0
        while not result["done"]:
            for seat in result["seats"]:
                env.act(seat, decide(env.observe(seat)))
                decisions += 1
            result = env.step()

        standings = env.standings()

    print(f"Tick {standings['tick']}, {decisions} decisions, {time.time() - start:.0f}s")
    for p in sorted(standings["players"], key=lambda p: -p["stars"]):
        tag = " (agent)" if p["agent"] else ""
        print(f"  {p['alias']:<12} stars={p['stars']:>3} ships={p['ships']:>6}{tag}")
    if standings["winner"] is not None:
        print("Winner:", standings["players"][standings["winner"]]["alias"])


if __name__ == "__main__":
    main()
