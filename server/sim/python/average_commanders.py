"""Average commander checkpoints into one ("model soup").

    python3 average_commanders.py <out.pt> <a.pt> <b.pt> [more.pt ...]

Every weight of the result is the mean of that weight in the inputs. This
works best for commanders trained from the same starting point (such as
two ppo_train.py runs from the same belief model, or two checkpoints of one
run); for unrelated ones the average is usually worse than either. Compare
it with its parents with ppo_train.py's evaluation, or add it to a league.
"""

import sys

import torch


def average(paths):
    saved = [torch.load(p, map_location="cpu") for p in paths]
    techs = {s["n_techs"] for s in saved}
    if len(techs) != 1:
        raise ValueError("The checkpoints are for different technology lists.")
    states = [s["model"] for s in saved]
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(s[k] for s in states) / len(states)
        else:
            out[k] = v
    return {"model": out, "n_techs": techs.pop()}


if __name__ == "__main__":
    torch.save(average(sys.argv[2:]), sys.argv[1])
    print(f"averaged {len(sys.argv) - 2} checkpoints -> {sys.argv[1]}")
