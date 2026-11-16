"""Flawed built-in AI opponents, for training and evaluating against the
kinds of mistakes people make.

Each persona is a set of AiConfig overrides (services/types/Ai.ts) for a
built-in AI seat, with its strength drawn from a range so the commander
learns to punish a kind of mistake rather than one opponent's habits.

TRAIN personas appear in league games and in evaluation set C. HELD_OUT
personas never appear in training; evaluation set D uses them to check
that what the commander learned carries over to mistakes it never saw.
"""

import numpy as np


def _overextender(rng):
    # Grabs and attacks before defending, with thin margins, far away.
    return {"orderPriorities": {"defend": 2, "invade": 4, "claim": 3},
            "invasionAttackFactor": float(rng.uniform(1.0, 1.2)),
            "orderReachCycles": int(rng.integers(3, 5))}


def _hoarder(rng):
    # Lets ships pile up behind the lines; the border stays thin.
    return {"logisticStockpileCycles": float(rng.uniform(2.5, 5.0))}


def _turtle(rng):
    # Rarely attacks and only nearby: its neighbours grow unchallenged.
    return {"invasionAttackFactor": float(rng.uniform(5.0, 8.0)), "orderReachCycles": 1}


def _sloppy(rng):
    # Overlooks a share of its orders each cycle and misjudges the rest:
    # stars left undefended, parts of the frontier forgotten.
    return {"orderDropout": float(rng.uniform(0.2, 0.45)),
            "orderNoise": float(rng.uniform(0.3, 0.9))}


def _absent(rng):
    # Checks in only every few ticks: slow to react, spending late.
    return {"actEveryTicks": int(rng.integers(4, 13))}


def _careless(rng):
    # Sizes attacks on ship counts alone, ignoring weapons and the
    # defender bonus, with little margin.
    return {"ignoreDefenderWeapons": True,
            "invasionAttackFactor": float(rng.uniform(1.0, 1.3))}


TRAIN = {"overextender": _overextender, "hoarder": _hoarder, "turtle": _turtle,
         "sloppy": _sloppy}
HELD_OUT = {"absent": _absent, "careless": _careless}
ALL = {**TRAIN, **HELD_OUT}


def sample(rng, pool=TRAIN):
    """A random persona from pool: (name, AiConfig overrides)."""
    name = str(rng.choice(sorted(pool)))
    return name, pool[name](rng)


def assign(rng, seats, pool=TRAIN):
    """A random persona for each of seats: {seat: (name, overrides)}."""
    return {int(s): sample(rng, pool) for s in seats}
