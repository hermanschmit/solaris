"""Factored PPO + Distributional Critic + Cross-Play League Trainer (`admiral_train.py`).

Trains `AdmiralNet` in `server/sim/admiral/` while co-playing against:
  1. Frontier self-play (`Admiral` recent EMA checkpoints),
  2. Historical PFSP archive + frozen `Commander` checkpoints (e.g. from
     `server/data/league1/ema_*.pt`), and
  3. Diverse `personas.py` & standard built-in AI seats,
and writes shared `SBD1` `.sbd` belief recordings usable by both flows.

Example usage:
    /home/herman/solaris-ml/bin/python3 sim/admiral/python/admiral_train.py \
        --run-dir data/admiral1 \
        --commander-dir data/league1 \
        --iters 200 --games-per-iter 24 --workers 8
"""

import argparse
import copy
import glob
import json
import os
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_UPSTREAM_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python")
)
if _UPSTREAM_PY not in sys.path:
    sys.path.insert(0, _UPSTREAM_PY)

import belief_data as bd  # noqa: E402
import personas as pa  # noqa: E402
import presets  # noqa: E402

from admiral_env import AdmiralEnv  # noqa: E402
from admiral_net import HEAD_GROUPS, AdmiralNet  # noqa: E402
from admiral_rollout import (  # noqa: E402
    _rollout_worker,
    evaluate_suite,
    load_admiral_model,
)
from belief_tracker import VISIBLE  # noqa: E402

# PPO & Critic Hyperparameters
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPS = 0.20
MAX_GROUP_KL = 0.045
LR = 2.5e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.98

# Per-group entropy weights (keep spatial & macro exploration healthy without noisy knobs)
ENTROPY_WEIGHTS = {
    "spatial": 0.008,
    "budget": 0.002,
    "knobs": 0.002,
    "macro": 0.010,
}
TRUTH_VALUE_COEF = 0.50
FOG_VALUE_COEF = 0.25
AUX_BELIEF_COEF = 0.15


def save_admiral_checkpoint(
    path: str | Path,
    model: AdmiralNet,
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
):
    payload = {
        "arch": "admiral",
        "iteration": int(iteration),
        "config": {
            "n_techs": model.n_techs,
            "d": model.d,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
        },
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def update_ema(ema_model: AdmiralNet, live_model: AdmiralNet, decay: float = EMA_DECAY):
    for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):
        p_ema.data.mul_(decay).add_(p_live.data.cpu(), alpha=1.0 - decay)
    for b_ema, b_live in zip(ema_model.buffers(), live_model.buffers()):
        b_ema.data.copy_(b_live.data.cpu())


class CompactCrossPlayLeague:
    """3-Tier opponent matchmaker preventing archive dilution while enabling
    continuous cross-play co-refinement against `Commander` checkpoints."""

    def __init__(self, league_file: Path, max_commander_imports: int = 8):
        self.league_file = league_file
        self.max_commander_imports = max_commander_imports
        self.members: dict[str, dict] = {}
        if league_file.exists():
            data = json.loads(league_file.read_text())
            self.members = data.get("members", {})

    def save(self):
        tmp = str(self.league_file) + ".tmp"
        Path(tmp).write_text(json.dumps({"members": self.members}, indent=2))
        os.replace(tmp, self.league_file)

    def register_member(self, name: str, path: str | Path, kind: str, step: int = 0):
        p_abs = os.path.abspath(str(path))
        if not os.path.exists(p_abs):
            return
        if name not in self.members:
            self.members[name] = {
                "path": p_abs,
                "kind": kind,
                "step": int(step),
                "games": 0,
                "wins_against": 0.0,  # Learner beat this member
            }
        else:
            self.members[name]["path"] = p_abs

    def import_commander_dir(self, commander_dir: str | Path | None):
        """Auto-discover `ema_*.pt` (or `ckpt_*.pt`) checkpoints from an existing
        Commander run directory (e.g. `server/data/league1`), selecting an evenly
        spaced subset plus the newest checkpoint so cross-play spans skill tiers."""
        if not commander_dir:
            return
        cdir = Path(commander_dir)
        if not cdir.is_dir():
            return

        def _num(p: str) -> int:
            m = re.search(r"(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else 0

        emas = sorted(glob.glob(str(cdir / "ema_*.pt")), key=_num)
        if not emas:
            emas = sorted(glob.glob(str(cdir / "ckpt_*.pt")), key=_num)
        if not emas:
            return

        if len(emas) > self.max_commander_imports:
            idxs = np.linspace(0, len(emas) - 1, self.max_commander_imports, dtype=int)
            emas = [emas[i] for i in sorted(set(idxs.tolist()))]

        for p in emas:
            step = _num(p)
            tag = f"commander_{cdir.name}_{os.path.splitext(os.path.basename(p))[0]}"
            self.register_member(tag, p, kind="commander", step=step)

    def sample_matchup(
        self,
        rng: np.random.Generator,
        learner_seats: list[int],
        n_opponent_seats: int,
    ) -> tuple[dict[int, dict], dict[int, dict]]:
        """Assign non-learner seats across:
        - Frontier recent Admiral EMAs (~45%),
        - PFSP historical Admiral + imported Commander cross-play (~40%),
        - Persona / default built-in AIs (remaining seats)."""
        avail_seats = [
            s for s in range(presets.PLAYERS) if s not in learner_seats
        ]
        rng.shuffle(avail_seats)

        valid_members = {
            k: v for k, v in self.members.items() if os.path.exists(v["path"])
        }
        opponents: dict[int, dict] = {}
        ai_configs: dict[int, dict] = {}

        if valid_members and n_opponent_seats > 0:
            adm_members = sorted(
                [(k, v) for k, v in valid_members.items() if v["kind"] == "admiral"],
                key=lambda kv: kv[1].get("step", 0),
                reverse=True,
            )
            frontier = adm_members[:3]
            archive = list(valid_members.items())

            # PFSP weights with Archive-Size-Scaled Floor (fixes 108-member dilution!)
            floor = 0.15 / max(1, len(archive))
            pfsp_w = []
            for _, info in archive:
                g = max(1, info.get("games", 0))
                learner_wr = (
                    info.get("wins_against", 0.5) / float(g)
                    if info.get("games", 0) > 0
                    else 0.35
                )
                # Focus on opponents that still beat or tie the learner
                pfsp_w.append(max(floor, (1.0 - learner_wr) ** 2 + floor))
            pfsp_probs = np.asarray(pfsp_w, dtype=np.float64)
            pfsp_probs /= pfsp_probs.sum()

            chosen_seats = avail_seats[: min(n_opponent_seats, len(avail_seats))]
            for s in chosen_seats:
                roll = float(rng.random())
                if frontier and roll < 0.45:
                    name, info = frontier[int(rng.integers(0, len(frontier)))]
                else:
                    idx = int(rng.choice(len(archive), p=pfsp_probs))
                    name, info = archive[idx]
                opponents[int(s)] = {
                    "name": name,
                    "kind": info["kind"],
                    "path": info["path"],
                }

        # Remaining non-agent seats: mix TRAIN persona-tuned AIs (40% of games) & standard AI
        rem_seats = [s for s in avail_seats if s not in opponents]
        if rem_seats and float(rng.random()) < 0.40:
            for s, (_, cfg) in pa.assign(rng, rem_seats, pa.TRAIN).items():
                ai_configs[int(s)] = cfg

        return opponents, ai_configs

    def update_outcomes(
        self,
        opponents: dict[int, dict],
        learner_seats: list[int],
        place_all: np.ndarray,
    ):
        """Update PFSP win-rates from pairwise placement comparisons."""
        if not opponents or len(learner_seats) == 0:
            return
        learner_mean_place = float(np.mean([place_all[s] for s in learner_seats]))
        for s, spec in opponents.items():
            name = spec.get("name")
            if name not in self.members:
                continue
            opp_place = float(place_all[int(s)])
            score = (
                1.0
                if learner_mean_place < opp_place
                else (0.5 if learner_mean_place == opp_place else 0.0)
            )
            self.members[name]["games"] += 1
            self.members[name]["wins_against"] += score


def compute_auxiliary_belief_loss(
    belief_logits: torch.Tensor,
    aux_owner: torch.Tensor,
    aux_vis: torch.Tensor,
    aux_true_ships: torch.Tensor,
    aux_true_infra: torch.Tensor,
    n_players: int,
) -> torch.Tensor:
    """Conservation-consistent KL loss on hidden enemy stars:
    trains `belief_logits` (`[B, S, 4]`) to predict the exact fraction of each
    opponent's unseen ships and unseen infrastructure sitting at each hidden star!"""
    B = belief_logits.shape[0]
    total_loss = belief_logits.new_zeros(())
    count = 0

    for b in range(B):
        own_b = aux_owner[b]
        vis_b = aux_vis[b]
        for q in range(n_players):
            # Hidden ships of player q
            h_s = (own_b == q) & (vis_b != VISIBLE)
            if int(h_s.sum()) >= 2:
                t_sh = aux_true_ships[b, h_s].clamp(min=0.0)
                tot = t_sh.sum()
                if tot > 2.0:
                    target_p = t_sh / tot
                    log_q = F.log_softmax(belief_logits[b, h_s, 0], dim=0)
                    total_loss = total_loss - (target_p * log_q).sum()
                    count += 1

            # Hidden industry of player q (channel 2 of belief_logits, channel 1 of mem_infra)
            h_i = (own_b == q) & (vis_b == 0)
            if int(h_i.sum()) >= 2:
                t_ind = aux_true_infra[b, h_i, 1].clamp(min=0.0)
                tot_i = t_ind.sum()
                if tot_i > 1.0:
                    target_pi = t_ind / tot_i
                    log_qi = F.log_softmax(belief_logits[b, h_i, 2], dim=0)
                    total_loss = total_loss - (target_pi * log_qi).sum()
                    count += 1

    if count == 0:
        return total_loss
    return total_loss / float(count)


def build_dataset_from_rollouts(
    npz_files: list[Path],
    model: AdmiralNet,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load rollout `.npz` files, evaluate `truth_critic` along each seat's
    trajectory, and compute GAE advantages + distributional rank targets."""
    model.eval()

    in_lists: dict[str, list[np.ndarray]] = {}
    act_lists: dict[str, list[np.ndarray]] = {}
    truth_lists: dict[str, list[np.ndarray]] = {}
    old_logp_groups = {g: [] for g in HEAD_GROUPS}
    adv_list: list[np.ndarray] = []
    target_rank_list: list[np.ndarray] = []
    bonus_ret_list: list[np.ndarray] = []
    aux_owner_list: list[np.ndarray] = []
    aux_vis_list: list[np.ndarray] = []
    aux_ships_list: list[np.ndarray] = []
    aux_infra_list: list[np.ndarray] = []

    in_keys = (
        "star_x",
        "edge_x",
        "player_x",
        "owner_rel",
        "time_x",
        "own_mask",
        "non_own_mask",
        "free_own_mask",
        "research_mask",
        "spec_mask",
    )
    truth_keys = ("star_x", "edge_x", "player_x", "owner_rel", "time_x")
    act_keys = (
        "att1",
        "att2",
        "def",
        "ind",
        "sci",
        "wg",
        "budget",
        "knobs",
        "research",
        "spec",
        "spec_star",
    )

    with torch.no_grad():
        for fpath in npz_files:
            data = np.load(fpath)
            if len(data["seat"]) == 0:
                continue

            seats_arr = data["seat"]
            bonuses = data["bonus"].astype(np.float32)
            agents = data["agents"].tolist()
            places = data["place"].tolist()
            final_rewards = data["final_reward"].astype(np.float32).tolist()

            # Evaluate privileged truth critic across all steps in this game
            t_batch = {
                k: torch.from_numpy(data["truth_" + k]).to(device)
                for k in truth_keys
            }
            _, truth_vals_t, _ = model.forward_truth_critic(t_batch)
            truth_vals = truth_vals_t.cpu().numpy()

            N = len(seats_arr)
            gae_adv = np.zeros(N, dtype=np.float32)
            bonus_ret = np.zeros(N, dtype=np.float32)
            rank_targets = np.zeros(N, dtype=np.int64)

            for s_idx, s_id in enumerate(agents):
                idxs = np.flatnonzero(seats_arr == s_id)
                if len(idxs) == 0:
                    continue
                rank_targets[idxs] = int(places[s_idx]) - 1
                fin_r = float(final_rewards[s_idx])

                last_gae = 0.0
                running_bonus = 0.0
                for pos in range(len(idxs) - 1, -1, -1):
                    t_i = idxs[pos]
                    is_last = pos == len(idxs) - 1
                    r_t = float(bonuses[t_i]) + (fin_r if is_last else 0.0)
                    next_v = 0.0 if is_last else float(truth_vals[idxs[pos + 1]])
                    delta = r_t + (0.0 if is_last else GAMMA * next_v) - float(truth_vals[t_i])
                    last_gae = delta + GAMMA * GAE_LAMBDA * (0.0 if is_last else last_gae)
                    gae_adv[t_i] = last_gae

                    running_bonus = float(bonuses[t_i]) + GAMMA * running_bonus
                    bonus_ret[t_i] = np.clip(running_bonus, -0.5, 0.5)

            for k in in_keys:
                in_lists.setdefault(k, []).append(data["in_" + k])
            for k in act_keys:
                act_lists.setdefault(k, []).append(data["act_" + k])
            for k in truth_keys:
                truth_lists.setdefault(k, []).append(data["truth_" + k])
            for g in HEAD_GROUPS:
                old_logp_groups[g].append(data["logp_" + g].astype(np.float32))

            adv_list.append(gae_adv)
            target_rank_list.append(rank_targets)
            bonus_ret_list.append(bonus_ret)
            aux_owner_list.append(data["aux_owner"])
            aux_vis_list.append(data["aux_vis"])
            aux_ships_list.append(data["aux_true_ships"])
            aux_infra_list.append(data["aux_true_infra"])

    dataset: dict[str, torch.Tensor] = {}
    for k, v in in_lists.items():
        dataset["in_" + k] = torch.from_numpy(np.concatenate(v, axis=0))
    for k, v in act_lists.items():
        dataset["act_" + k] = torch.from_numpy(np.concatenate(v, axis=0))
    for k, v in truth_lists.items():
        dataset["truth_" + k] = torch.from_numpy(np.concatenate(v, axis=0))
    for g in HEAD_GROUPS:
        dataset["old_logp_" + g] = torch.from_numpy(
            np.concatenate(old_logp_groups[g], axis=0)
        )

    adv_all = torch.from_numpy(np.concatenate(adv_list, axis=0))
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-6)
    dataset["adv"] = adv_all
    dataset["target_rank"] = torch.from_numpy(np.concatenate(target_rank_list, axis=0))
    dataset["bonus_ret"] = torch.from_numpy(np.concatenate(bonus_ret_list, axis=0))
    dataset["aux_owner"] = torch.from_numpy(np.concatenate(aux_owner_list, axis=0))
    dataset["aux_vis"] = torch.from_numpy(np.concatenate(aux_vis_list, axis=0))
    dataset["aux_ships"] = torch.from_numpy(np.concatenate(aux_ships_list, axis=0))
    dataset["aux_infra"] = torch.from_numpy(np.concatenate(aux_infra_list, axis=0))
    return dataset


def ppo_update(
    model: AdmiralNet,
    optimizer: torch.optim.Optimizer,
    dataset: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 64,
) -> dict[str, float]:
    """Factored Per-Head PPO + Dual Distributional Critic + Aux Belief Update."""
    model.train()
    N = dataset["adv"].shape[0]
    n_players = dataset["in_player_x"].shape[1]

    in_keys = [k[3:] for k in dataset if k.startswith("in_")]
    act_keys = [k[4:] for k in dataset if k.startswith("act_")]
    truth_keys = [k[6:] for k in dataset if k.startswith("truth_")]

    stats_acc: dict[str, list[float]] = {
        "policy_loss": [],
        "truth_val_loss": [],
        "fog_val_loss": [],
        "belief_loss": [],
        "entropy": [],
    }
    for g in HEAD_GROUPS:
        stats_acc[f"kl_{g}"] = []
        stats_acc[f"clip_{g}"] = []

    early_stopped = False
    for _ in range(epochs):
        if early_stopped:
            break
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue

            b_in = {k: dataset["in_" + k][idx].to(device) for k in in_keys}
            b_act = {k: dataset["act_" + k][idx].to(device) for k in act_keys}
            b_truth = {k: dataset["truth_" + k][idx].to(device) for k in truth_keys}
            adv = dataset["adv"][idx].to(device)
            target_rank = dataset["target_rank"][idx].to(device)
            bonus_ret = dataset["bonus_ret"][idx].to(device)

            eval_out = model.act_or_evaluate(b_in, actions=b_act, sample=False)
            p_out = eval_out["policy_out"]

            # 1. Factored Per-Head PPO Surrogate Objective
            pol_loss = adv.new_zeros(())
            ent_bonus = adv.new_zeros(())
            group_kls = {}

            for g in HEAD_GROUPS:
                new_lp = eval_out["logp_groups"][g]
                old_lp = dataset["old_logp_" + g][idx].to(device)
                log_ratio = (new_lp - old_lp).clamp(-15.0, 15.0)
                ratio = log_ratio.exp()
                approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                clip_frac = float(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())

                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
                pol_loss = pol_loss - torch.min(surr1, surr2).mean()
                ent_bonus = ent_bonus + ENTROPY_WEIGHTS[g] * eval_out["entropy_groups"][g].mean()

                group_kls[g] = approx_kl
                stats_acc[f"kl_{g}"].append(approx_kl)
                stats_acc[f"clip_{g}"].append(clip_frac)

            if any(kl > MAX_GROUP_KL for kl in group_kls.values()):
                early_stopped = True
                break

            # 2. Privileged Truth Distributional Critic + Fog Search Distributional Critic
            t_rank_logits, t_total_val, t_place_val = model.forward_truth_critic(b_truth)
            t_bonus_pred = t_total_val - t_place_val
            truth_val_loss = F.cross_entropy(t_rank_logits, target_rank) + F.smooth_l1_loss(
                t_bonus_pred, bonus_ret
            )

            f_rank_logits = p_out["fog_rank_logits"]
            f_bonus_pred = p_out["fog_value"] - p_out["fog_place_val"]
            fog_val_loss = F.cross_entropy(f_rank_logits, target_rank) + F.smooth_l1_loss(
                f_bonus_pred, bonus_ret
            )

            # 3. Auxiliary Conservation-Belief Loss
            belief_loss = compute_auxiliary_belief_loss(
                p_out["belief_logits"],
                dataset["aux_owner"][idx].to(device),
                dataset["aux_vis"][idx].to(device),
                dataset["aux_ships"][idx].to(device),
                dataset["aux_infra"][idx].to(device),
                n_players=n_players,
            )

            total_loss = (
                pol_loss
                - ent_bonus
                + TRUTH_VALUE_COEF * truth_val_loss
                + FOG_VALUE_COEF * fog_val_loss
                + AUX_BELIEF_COEF * belief_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            stats_acc["policy_loss"].append(float(pol_loss.item()))
            stats_acc["truth_val_loss"].append(float(truth_val_loss.item()))
            stats_acc["fog_val_loss"].append(float(fog_val_loss.item()))
            stats_acc["belief_loss"].append(float(belief_loss.item()))
            stats_acc["entropy"].append(float(eval_out["entropy"].mean().item()))

    model.eval()
    return {
        k: round(float(np.mean(v)), 5) if v else 0.0
        for k, v in stats_acc.items()
    }


def train(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    belief_dir = run_dir / "belief"
    belief_dir.mkdir(exist_ok=True)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Probe technology count once
    with AdmiralEnv() as probe_env:
        probe_env.reset(admiral_seats=[0], settings=presets.NAME, seed=1, max_ticks=5)
        n_techs = len(probe_env.header()["technologies"])

    # Resume latest checkpoint if present
    existing_ckpts = sorted(
        run_dir.glob("ckpt_*.pt"),
        key=lambda p: int(re.search(r"(\d+)\.pt$", p.name).group(1)),
    )
    start_iter = 0
    if existing_ckpts:
        latest = existing_ckpts[-1]
        start_iter = int(re.search(r"(\d+)\.pt$", latest.name).group(1))
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model = load_admiral_model(n_techs, latest, d=args.d_model, n_layers=args.n_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        ema_path = run_dir / f"ema_{start_iter}.pt"
        ema_model = load_admiral_model(
            n_techs, ema_path if ema_path.exists() else latest, d=args.d_model, n_layers=args.n_layers
        )
    else:
        model = AdmiralNet(n_techs, d=args.d_model, n_layers=args.n_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        ema_model = copy.deepcopy(model).cpu()
        save_admiral_checkpoint(run_dir / "ckpt_0.pt", model, optimizer, 0)
        save_admiral_checkpoint(run_dir / "ema_0.pt", ema_model, None, 0)

    league = CompactCrossPlayLeague(run_dir / "league.json")
    league.import_commander_dir(args.commander_dir)
    for extra_c in args.commander_ckpt or []:
        if os.path.exists(extra_c):
            league.register_member(
                f"commander_{os.path.basename(extra_c)}", extra_c, kind="commander"
            )
    league.save()

    rng = np.random.default_rng(args.seed + start_iter * 1000)
    log_path = run_dir / "log.jsonl"
    eval_path = run_dir / "eval.jsonl"

    pool = Pool(args.workers) if args.workers > 1 else None
    try:
        for it in range(start_iter + 1, start_iter + args.iters + 1):
            t0 = time.time()
            roll_dir = run_dir / f"rollout_{it}"
            roll_dir.mkdir(exist_ok=True)
            cur_ckpt = run_dir / f"ckpt_{it - 1}.pt"
            if not cur_ckpt.exists():
                save_admiral_checkpoint(cur_ckpt, model, optimizer, it - 1)

            jobs = []
            job_meta = {}
            base_seed = args.seed + it * 10_000
            n_opp = presets.PRESET["league_opponents"]

            for g_idx in range(args.games_per_iter):
                g_seed = base_seed + g_idx
                g_rng = np.random.default_rng(g_seed)
                l_seats = sorted(
                    int(s)
                    for s in g_rng.choice(
                        presets.PLAYERS, presets.SEATS, replace=False
                    )
                )
                opps, ai_cfgs = league.sample_matchup(rng, l_seats, n_opp)
                rec_file = (
                    str(belief_dir / f"i{it}_s{g_seed}.sbd")
                    if args.record_every > 0 and (g_idx % args.record_every == 0)
                    else None
                )
                job_meta[g_seed] = (l_seats, opps)
                jobs.append(
                    (
                        str(roll_dir),
                        g_seed,
                        presets.SEATS,
                        str(cur_ckpt),
                        args.max_ticks,
                        opps,
                        rec_file,
                        ai_cfgs,
                        True,
                    )
                )

            if pool is not None:
                list(pool.imap_unordered(_rollout_worker, jobs))
            else:
                for j in jobs:
                    _rollout_worker(j)

            npz_files = sorted(roll_dir.glob("r*.npz"))
            places_iter = []
            ticks_iter = []
            for f in npz_files:
                d = np.load(f)
                g_seed = int(d["seed"])
                places_iter.extend(d["place"].tolist())
                ticks_iter.append(int(d["ticks"]))
                if g_seed in job_meta:
                    l_seats, opps = job_meta[g_seed]
                    league.update_outcomes(opps, l_seats, d["place_all"])

            dataset = build_dataset_from_rollouts(npz_files, model, device)
            upd_stats = ppo_update(
                model,
                optimizer,
                dataset,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            update_ema(ema_model, model, decay=EMA_DECAY)

            ckpt_out = run_dir / f"ckpt_{it}.pt"
            ema_out = run_dir / f"ema_{it}.pt"
            save_admiral_checkpoint(ckpt_out, model, optimizer, it)
            save_admiral_checkpoint(ema_out, ema_model, None, it)

            if it % args.snapshot_every == 0:
                league.register_member(f"admiral_ema_{it}", ema_out, kind="admiral", step=it)
            league.save()

            if not args.keep_rollouts:
                shutil.rmtree(roll_dir, ignore_errors=True)
                old_ckpt = run_dir / f"ckpt_{it - 2}.pt"
                if (it - 2) > 0 and (it - 2) % args.snapshot_every != 0 and old_ckpt.exists():
                    old_ckpt.unlink()

            dt = time.time() - t0
            entry = {
                "iter": it,
                "steps": int(dataset["adv"].shape[0]),
                "mean_place": round(float(np.mean(places_iter)), 3),
                "win_rate": round(float(np.mean([p == 1 for p in places_iter])), 3),
                "mean_ticks": round(float(np.mean(ticks_iter)), 1),
                "league_size": len(league.members),
                "sec": round(dt, 1),
                **upd_stats,
            }
            with open(log_path, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

            if args.eval_every > 0 and it % args.eval_every == 0:
                eval_seeds = list(range(950_000, 950_000 + args.eval_games))
                ev_builtin = evaluate_suite(
                    eval_seeds,
                    checkpoint=str(ema_out),
                    workers=args.workers,
                    max_ticks=args.max_ticks,
                )
                ev_entry = {"iter": it, "vs_builtin": ev_builtin}
                # Also benchmark head-to-head against newest imported Commander if available
                cmd_members = [
                    v["path"]
                    for v in league.members.values()
                    if v["kind"] == "commander" and os.path.exists(v["path"])
                ]
                if cmd_members:
                    ev_cmd = evaluate_suite(
                        eval_seeds,
                        checkpoint=str(ema_out),
                        opponent=cmd_members[-1],
                        workers=args.workers,
                        max_ticks=args.max_ticks,
                    )
                    ev_entry["vs_commander"] = ev_cmd
                with open(eval_path, "a") as fp:
                    fp.write(json.dumps(ev_entry) + "\n")
                print("EVAL:", json.dumps(ev_entry), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def main():
    p = argparse.ArgumentParser(description="Train Admiral with Factored PPO & Cross-Play League.")
    p.add_argument("--run-dir", required=True, help="Output directory for checkpoints & logs.")
    p.add_argument(
        "--commander-dir",
        default=None,
        help="Optional existing Commander league dir (e.g. data/league1) for cross-play co-refinement.",
    )
    p.add_argument(
        "--commander-ckpt",
        action="append",
        default=[],
        help="Additional Commander .pt checkpoint(s) to include in cross-play league.",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-ticks", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--record-every", type=int, default=4, help="Save .sbd belief file every N games (0=off).")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=16)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--keep-rollouts", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
