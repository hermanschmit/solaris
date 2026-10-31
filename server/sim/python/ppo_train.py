"""Train the commander with PPO, against the built-in AI.

    python3 ppo_train.py <run_dir> [iterations] [games] [workers] [belief.pt|-] [eval_games]

Each iteration plays `games` games with the current commander in 3 random
seats (commander_rollout.py), learns from them, and saves the next
checkpoint:

  run_dir/ckpt_<i>.pt     the commander after i iterations (ckpt_0 is the
                          untrained one, which plays like the built-in AI)
  run_dir/iter_<i>/       the games played with ckpt_<i>
  run_dir/log.jsonl       one line per iteration: how the games went and
                          how training went
  run_dir/eval.jsonl      one line per evaluation

Every EVAL_EVERY iterations (and at the start) it evaluates: the commander
plays `eval_games` fixed seeds with its most likely decisions, and its
placements are compared with the built-in AI's in the same seats on the
same seeds, where the built-in AI sees the whole game.

It picks up where it left off: run it again with the same run_dir to
continue. The belief model initializes the commander's star encoding; it is
only used when starting.

Training runs on the GPU when there is one. Learning happens in the critic
and all heads, and, more slowly, in the star encoding.
"""

import json
import os
import queue
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import belief_data as bd
import commander_rollout as cr

GAMMA = 0.998  # per decision; a seat makes a few hundred decisions a game
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCH = 64
LR_HEADS = 1e-4
LR_TRUNK = 1e-5  # the belief-trained star encodings move more slowly
VALUE_COEF = 0.5
ENTROPY_COEF = 1e-3
MAX_GRAD = 0.5
TARGET_KL = 0.03  # stop an iteration's epochs early past this
MAX_LOG_RATIO = 20.0  # a decision's log-probability is a sum over many orders, so can move far
DUAL_CLIP = 3.0  # bounds the loss of a bad action the policy now likes much more
SEATS = 3
EVAL_EVERY = 5
EVAL_FIRST_SEED = 1  # evaluation seeds; training seeds start far above
TRAIN_FIRST_SEED = 1_000_000
MAX_TICKS = int(os.environ.get("PPO_MAX_TICKS", 3000))  # shorter games, for trying it out

POLICY_KEYS = ["star_x", "player_x", "owner", "time", "cand_star", "cand_kind",
               "cand_score", "cand_ticks", "cand_mask", "research_turn", "spec_mask",
               "spec_budget"]
TRUTH_KEYS = ["star_x", "player_x", "owner", "time"]


def ckpt_path(run_dir, i):
    return Path(run_dir) / f"ckpt_{i}.pt"


def latest(run_dir):
    i = 0
    while ckpt_path(run_dir, i + 1).exists():
        i += 1
    return i


def n_techs():
    from solaris_env import SolarisEnv
    with SolarisEnv() as env:
        env.reset(agents=[], seed=1, max_ticks=1)
        return len(bd.Static(env.header()).techs)


def train_seed(first, i, k):
    """The seed of the k-th game started while training iteration i."""
    return first + i * 1000 + k


class Collector:
    """Keeps every worker playing training games, so none waits for the
    slowest game of an iteration. An iteration learns from the first games
    to finish; games still playing then count towards the next one, though
    they were played with an earlier checkpoint (their recorded
    log-probabilities keep PPO correct). make_job(i, k) gives the arguments
    for job, the k-th game started in iteration i, and job returns the
    path of the game's record (and anything else, as a tuple)."""

    def __init__(self, pool, workers, job, make_job):
        self.pool, self.workers = pool, workers
        self.job, self.make_job = job, make_job
        self.finished = queue.Queue()
        self.playing = 0
        self.iteration, self.started = None, 0

    def _start_one(self):
        i, k = self.iteration, self.started
        self.started += 1
        self.playing += 1
        self.pool.apply_async(self.job, (self.make_job(i, k),),
                              callback=lambda r: self.finished.put((i, r)),
                              error_callback=lambda e: self.finished.put((i, e)))

    def start(self, i):
        """New games from now on use iteration i's checkpoint."""
        if i != self.iteration:
            self.iteration, self.started = i, 0
        while self.playing < self.workers:
            self._start_one()

    def take(self, n):
        """The next n games to finish, as (iteration started, result)."""
        out = []
        while len(out) < n:
            i, result = self.finished.get()
            self.playing -= 1
            if isinstance(result, BaseException):
                raise result
            out.append((i, result))
            self._start_one()
            print(f"  played {len(out)}/{n}", end="\r", flush=True)
        print()
        return out


def _train_job(args):
    out, seed = args[0], args[1]
    cr._job(args)
    return Path(out) / f"r{seed}.npz"


def train_jobs(run_dir):
    def make_job(i, k):
        out = Path(run_dir) / f"iter_{i}"
        out.mkdir(parents=True, exist_ok=True)
        return (str(out), train_seed(TRAIN_FIRST_SEED, i, k), SEATS, None,
                str(ckpt_path(run_dir, i)), MAX_TICKS)
    return make_job


class Steps:
    """Every decision of an iteration's games, with each agent seat's
    decisions together and in order, and its reward: the share-of-stars
    bonus at each, plus the placement reward at the seat's last."""

    def __init__(self, paths):
        parts = [np.load(p) for p in paths]
        C = max(z["in_cand_mask"].shape[1] for z in parts)
        cols, rewards, done, places = {}, [], [], []
        for z in parts:
            seat = z["seat"]
            order = np.lexsort((np.arange(len(seat)), seat))
            reward = z["bonus"].astype(np.float32)
            last = np.zeros(len(seat), bool)
            for s, r, p in zip(z["agents"], z["final_reward"], z["place"]):
                places.append(int(p))
                idx = np.nonzero(seat == s)[0]
                if len(idx):
                    reward[idx[-1]] += r
                    last[idx[-1]] = True
            rewards.append(reward[order])
            done.append(last[order])
            for k in z.files:
                if not (k.startswith("in_") or k.startswith("truth_") or k.startswith("act_")
                        or k == "logp"):
                    continue
                v = z[k][order]
                if (k.startswith("in_cand_") or k == "act_order") and v.shape[1] < C:
                    fill = -1 if k == "in_cand_star" else 0
                    v = np.concatenate(
                        [v, np.full((len(v), C - v.shape[1]), fill, v.dtype)], 1)
                cols.setdefault(k, []).append(v)
        self.cols = {k: torch.from_numpy(np.concatenate(v)) for k, v in cols.items()}
        self.reward = np.concatenate(rewards)
        self.done = np.concatenate(done)
        self.places = places
        self.hired = float(np.mean(np.concatenate([z["hired"] for z in parts]))) \
            if all("hired" in z.files for z in parts) else 0.0
        self.n = len(self.reward)

    def batch(self, idx, device):
        idx = torch.as_tensor(idx)
        get = lambda k: self.cols[k][idx].to(device)
        policy = {k: get("in_" + k) for k in POLICY_KEYS}
        policy["star_x"] = policy["star_x"].float()
        policy["cand_mask"] = policy["cand_mask"].bool()
        policy["spec_mask"] = policy["spec_mask"].bool()
        truth = {k: get("truth_" + k) for k in TRUTH_KEYS}
        act = {k[4:]: get(k) for k in self.cols if k.startswith("act_")}
        return policy, truth, act, get("logp").float()


@torch.no_grad()
def values(model, steps, device, size=256):
    out = []
    for s in range(0, steps.n, size):
        _, truth, _, _ = steps.batch(np.arange(s, min(s + size, steps.n)), device)
        out.append(model.value(truth).cpu().numpy())
    return np.concatenate(out)


def gae(reward, value, done):
    adv = np.zeros_like(value)
    last = 0.0
    for t in reversed(range(len(reward))):
        next_value = 0.0 if done[t] else value[t + 1]
        delta = reward[t] + GAMMA * next_value - value[t]
        last = delta + GAMMA * LAMBDA * (0.0 if done[t] else last)
        adv[t] = last
    return adv, adv + value


def optimizer(model):
    trunk = list(model.trunk.parameters())
    ids = {id(p) for p in trunk}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": trunk, "lr": LR_TRUNK},
                             {"params": rest, "lr": LR_HEADS}])


def update(model, opt, steps, device):
    v = values(model, steps, device)
    adv, ret = gae(steps.reward, v, steps.done)
    value_error = float(np.mean((ret - v) ** 2))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_t, ret_t = torch.from_numpy(adv).float(), torch.from_numpy(ret).float()

    stats = {"steps": steps.n, "value_error_before": value_error}
    sums, count = {}, 0
    for epoch in range(EPOCHS):
        kls = []
        perm = np.random.permutation(steps.n)
        for s in range(0, steps.n, MINIBATCH):
            idx = perm[s:s + MINIBATCH]
            policy, truth, act, old_logp = steps.batch(idx, device)
            a, r = adv_t[idx].to(device), ret_t[idx].to(device)

            out = model(policy)
            logp, ent = model.log_prob(out, policy, act)
            log_ratio = (logp - old_logp).clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
            ratio = torch.exp(log_ratio)
            objective = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
            # Dual-clip PPO: otherwise, for an action with negative advantage
            # whose probability has grown many times (as an older game's can
            # have), the loss grows without bound.
            objective = torch.where(a < 0, torch.max(objective, DUAL_CLIP * a), objective)
            pg = -objective.mean()
            value_loss = ((model.value(truth) - r) ** 2).mean()
            loss = pg + VALUE_COEF * value_loss - ENTROPY_COEF * ent.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            opt.step()

            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
                clipped = ((ratio - 1).abs() > CLIP).float().mean().item()
            kls.append(kl)
            for k, x in (("policy_loss", pg.item()), ("value_loss", value_loss.item()),
                         ("kl", kl), ("clip_frac", clipped)):
                sums[k] = sums.get(k, 0.0) + x
            count += 1
        stats["epochs"] = epoch + 1
        if np.mean(kls) > TARGET_KL:
            break
    stats.update({k: x / count for k, x in sums.items()})
    stats["order_std"] = model.order_log_std.exp().item()
    stats["config_std"] = model.config_log_std.exp().tolist()
    return stats


def _eval_job(args):
    seed, checkpoint, sample = args
    return seed, checkpoint, cr.play_eval(seed, SEATS, checkpoint, MAX_TICKS, sample)


def evaluate(run_dir, i, games, workers, pool=None):
    return {"iteration": i, **evaluate_checkpoint(
        ckpt_path(run_dir, i), Path(run_dir) / "eval_baseline.json", games, workers, pool)}


def evaluate_checkpoint(checkpoint, base_path, games, workers, pool=None, sample=False):
    """Placements of the commander's seats over the evaluation seeds,
    against the built-in AI (seeing everything) in the same seats. The
    built-in AI's games are kept in base_path, to be played only once.
    With a pool, its workers play them, once they finish the games they
    are playing. sample is how the commander decides (see
    CommanderAgent)."""
    seeds = list(range(EVAL_FIRST_SEED, EVAL_FIRST_SEED + games))
    base_path = Path(base_path)
    base = json.loads(base_path.read_text()) if base_path.exists() else {}
    jobs = [(s, str(checkpoint), sample) for s in seeds]
    jobs += [(s, None, False) for s in seeds if str(s) not in base]
    ours = {}
    own_pool = pool is None
    pool = pool or Pool(workers)
    try:
        for seed, checkpoint, places in pool.imap_unordered(_eval_job, jobs):
            if checkpoint:
                ours[seed] = list(places.values())
            else:
                base[str(seed)] = list(places.values())
    finally:
        if own_pool:
            pool.terminate()
    base_path.write_text(json.dumps(base))

    pairs = [(a, b) for s in seeds for a, b in zip(ours[s], base[str(s)])]
    mine, theirs = [a for a, _ in pairs], [b for _, b in pairs]
    return {"games": games, "placement": float(np.mean(mine)),
            "builtin_placement": float(np.mean(theirs)),
            "seats_better": sum(a < b for a, b in pairs),
            "seats_worse": sum(a > b for a, b in pairs),
            "wins": sum(p == 1 for p in mine), "builtin_wins": sum(p == 1 for p in theirs)}


def main():
    run_dir = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 52
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, os.cpu_count() - 1)
    belief = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    eval_games = int(sys.argv[6]) if len(sys.argv) > 6 else 26
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(run_dir, exist_ok=True)

    techs = n_techs()
    if not ckpt_path(run_dir, 0).exists():
        model = cr.build_model(techs, belief)
        torch.save({"model": model.state_dict(), "n_techs": techs}, ckpt_path(run_dir, 0))
    start = latest(run_dir)
    saved = torch.load(ckpt_path(run_dir, start), map_location="cpu")
    model = cr.build_model(techs)
    cr.load_commander(model, saved["model"])
    model.to(device)
    opt = optimizer(model)
    if "optimizer" in saved:
        try:
            opt.load_state_dict(saved["optimizer"])
        except ValueError:
            print("The checkpoint has fewer heads than the commander now: new optimizer state.")
    print(f"run {run_dir}: from iteration {start}, on {device}", flush=True)

    def log(name, record):
        with open(Path(run_dir) / name, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in record.items()), flush=True)

    with Pool(workers) as pool:
        collector = Collector(pool, workers, _train_job, train_jobs(run_dir))
        for i in range(start, start + iterations):
            if eval_games and i % EVAL_EVERY == 0:
                print(f"evaluating ckpt_{i}", flush=True)
                log("eval.jsonl", evaluate(run_dir, i, eval_games, workers, pool))

            t0 = time.time()
            print(f"iteration {i}: playing {games} games", flush=True)
            collector.start(i)
            played = collector.take(games)
            steps = Steps([path for _, path in played])
            t1 = time.time()
            model.eval()  # no dropout: the recorded log-probabilities must match
            stats = update(model, opt, steps, device)
            torch.save({"model": model.state_dict(), "n_techs": techs,
                        "optimizer": opt.state_dict()}, ckpt_path(run_dir, i + 1))
            log("log.jsonl", {"iteration": i, "placement": float(np.mean(steps.places)),
                              "hired": steps.hired,
                              "lag": float(np.mean([i - j for j, _ in played])), **stats,
                              "play_s": round(t1 - t0), "learn_s": round(time.time() - t1)})

        if eval_games:
            print(f"evaluating ckpt_{start + iterations}", flush=True)
            log("eval.jsonl", evaluate(run_dir, start + iterations, eval_games, workers, pool))


if __name__ == "__main__":
    main()
