"""Evaluate a commander checkpoint against the built-in AI.

    python3 commander_eval.py <commander.pt> [games] [workers] [baseline.json]

Plays the evaluation seeds ppo_train.py uses, with the commander's most
likely decisions under fog, and compares its seats' placements with the
built-in AI's in the same seats seeing the whole game. The built-in AI's
games are kept in baseline.json (by default next to the checkpoint), so
they are played once; a training run's eval_baseline.json works too.
"""

import os
import sys
from pathlib import Path

import ppo_train as pt

if __name__ == "__main__":
    checkpoint = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, os.cpu_count() - 1)
    base = sys.argv[4] if len(sys.argv) > 4 else Path(checkpoint).parent / "eval_baseline.json"
    result = pt.evaluate_checkpoint(checkpoint, base, games, workers)
    print(", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                    for k, v in result.items()))
