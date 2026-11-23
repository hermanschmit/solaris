"""Factored PPO + Distributional Critic + Cross-Play League Trainer (`admiral_train.py`).

Trains `AdmiralNet` in `server/sim/admiral/` while co-playing against:
  1. Frontier self-play (`Admiral` recent EMA checkpoints),
  2. Historical PFSP archive + frozen `Commander` checkpoints (e.g. from
     `server/data/league1/ema_*.pt`), and
  3. Diverse `personas.py` & standard built-in AI seats,
and writes shared `SBD1` `.sbd` belief recordings usable by both flows.

Example usage:
    python3 sim/admiral/python/admiral_train.py \
        --run-dir data/admiral3 \
        --pretrain data/admiral_belief.pt \
        --commander-dir data/league1 \
        --async-rollouts --iters 500 --games-per-iter 24 --workers 12
"""

import argparse
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
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
from admiral_net import HEAD_GROUPS, PART_NAMES, AdmiralNet  # noqa: E402
from admiral_rollout import (  # noqa: E402
    _async_rollout_worker,
    _rollout_worker,
    evaluate_suite,
    load_admiral_model,
)
from belief_tracker import VISIBLE  # noqa: E402

# PPO & Critic Hyperparameters
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPS = 0.20
DUAL_CLIP = 3.0
MAX_LOG_RATIO = 20.0
TARGET_KL = 0.045
LR_TRUNK = 1.5e-5
LR_HEADS = 8.0e-5
GRAD_CLIP = 0.5
EMA_DECAY = 0.85

# Per-group entropy weights
ENTROPY_WEIGHTS = {
    "spatial": 0.0015,
    "budget": 0.0015,
    "knobs": 0.0015,
    "macro": 0.004,
}
TRUTH_VALUE_COEF = 0.50
FOG_VALUE_COEF = 0.25
AUX_BELIEF_COEF = 0.15


def make_admiral_optimizer(
    model: AdmiralNet, lr_trunk: float = LR_TRUNK, lr_heads: float = LR_HEADS
) -> torch.optim.Optimizer:
    trunk_params = model.trunk_parameters()
    trunk_ids = {id(p) for p in trunk_params}
    head_params = [p for p in model.parameters() if id(p) not in trunk_ids]
    return torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": float(lr_trunk)},
            {"params": head_params, "lr": float(lr_heads)},
        ],
        weight_decay=1e-4,
    )


def load_pretrained_backbone(model: AdmiralNet, pretrain_path: str | Path) -> list[str]:
    """Load pre-trained `backbone`, `critic_backbone`, `belief_head`, `fog_critic`,
    and `truth_critic` weights from `admiral_pretrain.py` while leaving policy
    heads at their exact zero-initialized iter-0 priors."""
    saved = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    src_state = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    prefixes = (
        "backbone.",
        "critic_backbone.",
        "belief_head.",
        "fog_critic.",
        "truth_critic.",
        "aux_star_threat_head.",
        "aux_opp_state_head.",
        "aux_truth_expand_head.",
    )
    cur_state = model.state_dict()
    loaded_keys = []
    for k, v in src_state.items():
        if k.startswith(prefixes) and k in cur_state and cur_state[k].shape == v.shape:
            cur_state[k] = v.to(cur_state[k].device, dtype=cur_state[k].dtype)
            loaded_keys.append(k)
    model.load_state_dict(cur_state, strict=True)
    return loaded_keys


def save_admiral_checkpoint(
    path: str | Path,
    model: AdmiralNet,
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
):
    payload = {
        "arch": "admiral_v3",
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
        max_commander_seats: int = 2,
    ) -> tuple[dict[int, dict], dict[int, dict]]:
        """Assign non-learner seats across:
        - Frontier recent Admiral EMAs (~45%),
        - Variance-peaked PFSP (`f_var`, peaking around ~42% win rate so the
          learner climbs the Commander/Admiral ladder step-by-step rather than
          getting drowned by 92%-win-rate sharks before it learns to close games),
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
            cmd_members = [
                (k, v) for k, v in valid_members.items() if v["kind"] == "commander"
            ]
            frontier = adm_members[:3]

            # Adaptive Commander cross-play cap: when Learner win-rate vs Commander
            # is still <25%, limit to 1 Commander shark per lobby so Learner gets
            # rich +1.0 winning trajectories while climbing from weakest Commander up.
            if cmd_members:
                cmd_wrs = [
                    v.get("wins_against", 0.0) / max(1.0, float(v.get("games", 1)))
                    for _, v in cmd_members
                    if v.get("games", 0) > 0
                ]
                mean_cmd_wr = float(np.mean(cmd_wrs)) if cmd_wrs else 0.0
                eff_cmd_cap = (
                    1
                    if mean_cmd_wr < 0.25
                    else (max_commander_seats if mean_cmd_wr < 0.45 else n_opponent_seats)
                )
                eff_cmd_cap = min(eff_cmd_cap, max_commander_seats)
            else:
                eff_cmd_cap = 0

            def _pfsp_probs(items: list[tuple[str, dict]]) -> np.ndarray:
                floor = 0.12 / max(1, len(items))
                weights = []
                for _, info in items:
                    g = info.get("games", 0)
                    wr = info.get("wins_against", 0.0) / float(g) if g > 0 else 0.40
                    # Bell-shaped PFSP centered at wr=0.42 ("just barely beating us")
                    bell = float(np.exp(-(((wr - 0.42) / 0.22) ** 2)))
                    weights.append(max(floor, bell + floor))
                arr = np.asarray(weights, dtype=np.float64)
                return arr / arr.sum()

            archive_all = list(valid_members.items())
            probs_all = _pfsp_probs(archive_all)
            probs_adm = _pfsp_probs(adm_members) if adm_members else None

            cmd_picked = 0
            chosen_seats = avail_seats[: min(n_opponent_seats, len(avail_seats))]
            for s in chosen_seats:
                roll = float(rng.random())
                if frontier and roll < 0.45:
                    name, info = frontier[int(rng.integers(0, len(frontier)))]
                elif cmd_picked >= eff_cmd_cap and adm_members and probs_adm is not None:
                    idx = int(rng.choice(len(adm_members), p=probs_adm))
                    name, info = adm_members[idx]
                elif cmd_picked >= eff_cmd_cap and not adm_members:
                    # Leave remaining seat to built-in / persona AI during early iters 1..4
                    continue
                else:
                    idx = int(rng.choice(len(archive_all), p=probs_all))
                    name, info = archive_all[idx]

                if info["kind"] == "commander":
                    cmd_picked += 1
                opponents[int(s)] = {
                    "name": name,
                    "kind": info["kind"],
                    "path": info["path"],
                }

        # Remaining non-agent seats: mix TRAIN persona-tuned AIs (45% of games) & standard AI
        rem_seats = [s for s in avail_seats if s not in opponents]
        if rem_seats and float(rng.random()) < 0.45:
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


def _build_gpu_edge_x(
    edge_static_gpu: torch.Tensor,
    edge_idx: torch.Tensor,
    star_x: torch.Tensor,
    owner_rel: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct `[B, S, S, 5]` pairwise star edge features directly in GPU VRAM
    from compact `[K_total, S, S, 3]` float16 static/range slices + `star_x`/`owner_rel`."""
    e012 = edge_static_gpu.index_select(0, edge_idx).to(star_x.dtype)
    wg = (star_x[:, :, 14] > 0.5).to(star_x.dtype)
    fast_link = (wg.unsqueeze(2) * wg.unsqueeze(1)).unsqueeze(-1)
    same_own = (
        (owner_rel.unsqueeze(2) == owner_rel.unsqueeze(1))
        & (owner_rel.unsqueeze(2) >= 0)
    ).to(star_x.dtype).unsqueeze(-1)
    return torch.cat([e012, fast_link, same_own], dim=-1)


def compute_auxiliary_belief_loss(
    belief_logits: torch.Tensor,
    aux_owner: torch.Tensor,
    aux_vis: torch.Tensor,
    aux_true_ships: torch.Tensor,
    aux_true_infra: torch.Tensor,
    n_players: int,
    star_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fully vectorized `[B, P, S]` GPU conservation-consistent KL loss across
    all 4 hidden channels (`ships`, `economy`, `industry`, `science`), training
    `belief_logits` as exact additive log-residuals on top of `BeliefTracker`."""
    B, S, _ = belief_logits.shape
    dev = belief_logits.device
    dtype = belief_logits.dtype
    p_ids = torch.arange(n_players, device=dev, dtype=aux_owner.dtype).view(1, n_players, 1)
    own_bp = aux_owner.unsqueeze(1) == p_ids  # [B, P, S]

    total_num = belief_logits.new_zeros(())
    total_den = belief_logits.new_zeros(())

    # 1. Hidden ships (`vis != VISIBLE`)
    h_ships = own_bp & (aux_vis.unsqueeze(1) != VISIBLE)
    t_sh = torch.where(h_ships, aux_true_ships.unsqueeze(1).to(dtype).clamp(min=0.0), 0.0)
    tot_s = t_sh.sum(dim=-1)
    valid_s = (h_ships.sum(dim=-1) >= 2) & (tot_s > 2.0)
    target_ps = t_sh / tot_s.clamp(min=1e-6).unsqueeze(-1)
    raw_s = belief_logits[:, :, 0]
    if star_x is not None:
        prior_s = torch.expm1(star_x[:, :, 10] * 6.0).clamp(min=0.25).log()
        raw_s = raw_s + prior_s
    logits_s = (
        raw_s
        .unsqueeze(1)
        .expand(B, n_players, S)
        .masked_fill(~h_ships, -1e9)
    )
    log_qs = F.log_softmax(logits_s, dim=-1)
    ce_s = -(target_ps * torch.where(h_ships, log_qs, 0.0)).sum(dim=-1)
    total_num = total_num + torch.where(valid_s, ce_s, 0.0).sum()
    total_den = total_den + valid_s.to(dtype).sum()

    # 2. Hidden economy, industry, science (`vis == 0` HIDDEN)
    h_inf = own_bp & (aux_vis.unsqueeze(1) == 0)
    for ch_idx, logit_ch in ((0, 1), (1, 2), (2, 3)):
        t_inf = torch.where(
            h_inf,
            aux_true_infra[:, :, ch_idx].unsqueeze(1).to(dtype).clamp(min=0.0),
            0.0,
        )
        tot_inf = t_inf.sum(dim=-1)
        valid_inf = (h_inf.sum(dim=-1) >= 2) & (tot_inf > 1.0)
        target_pi = t_inf / tot_inf.clamp(min=1e-6).unsqueeze(-1)
        raw_i = belief_logits[:, :, logit_ch]
        if star_x is not None:
            prior_i = torch.expm1(star_x[:, :, 11 + ch_idx] * 3.5).clamp(min=0.10).log()
            raw_i = raw_i + prior_i
        logits_i = (
            raw_i
            .unsqueeze(1)
            .expand(B, n_players, S)
            .masked_fill(~h_inf, -1e9)
        )
        log_qi = F.log_softmax(logits_i, dim=-1)
        ce_i = -(target_pi * torch.where(h_inf, log_qi, 0.0)).sum(dim=-1)
        total_num = total_num + torch.where(valid_inf, ce_i, 0.0).sum()
        total_den = total_den + valid_inf.to(dtype).sum()

    return total_num / total_den.clamp(min=1.0)


def _dual_clipped_objective(log_ratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """PPO clipped surrogate with `DUAL_CLIP` (`ppo_train.py`) so negative-advantage
    off-policy actions from async games cannot produce unbounded loss spikes."""
    a = adv.view(-1, *([1] * (log_ratio.dim() - 1)))
    ratio = log_ratio.exp()
    obj = torch.min(ratio * a, ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * a)
    return torch.where(a < 0, torch.max(obj, DUAL_CLIP * a), obj)


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
    edge_tables: list[np.ndarray] = []
    step_edge_indices: list[np.ndarray] = []
    edge_offset = 0
    legacy_in_edge: list[np.ndarray] = []
    legacy_truth_edge: list[np.ndarray] = []

    old_logp_groups = {g: [] for g in HEAD_GROUPS}
    old_lp_parts = {p_name: [] for p_name in PART_NAMES}
    adv_list: list[np.ndarray] = []
    target_rank_list: list[np.ndarray] = []
    bonus_ret_list: list[np.ndarray] = []
    aux_owner_list: list[np.ndarray] = []
    aux_vis_list: list[np.ndarray] = []
    aux_ships_list: list[np.ndarray] = []
    aux_infra_list: list[np.ndarray] = []

    in_keys = (
        "star_x",
        "player_x",
        "tech_x",
        "owner_rel",
        "time_x",
        "own_mask",
        "non_own_mask",
        "frontier_mask",
        "free_own_mask",
        "alive_enemy_mask",
        "research_mask",
        "spec_mask",
    )
    truth_keys = ("star_x", "player_x", "owner_rel", "time_x")
    act_keys = (
        "frontier",
        "posture",
        "target_player",
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
            N = len(seats_arr)

            has_compact_edge = "edge_static" in data
            if has_compact_edge:
                g_edge_gpu = torch.from_numpy(data["edge_static"]).to(device)
                g_eidx_gpu = torch.from_numpy(
                    data["step_edge_idx"].astype(np.int64)
                ).to(device)
                edge_tables.append(data["edge_static"])
                step_edge_indices.append(
                    data["step_edge_idx"].astype(np.int64) + edge_offset
                )
                edge_offset += int(data["edge_static"].shape[0])
            else:
                legacy_in_edge.append(data["in_edge_x"])
                legacy_truth_edge.append(data["truth_edge_x"])

            # Evaluate privileged truth critic in bounded chunks
            truth_vals_chunks = []
            for cs in range(0, N, 128):
                ce = min(N, cs + 128)
                t_batch = {
                    k: torch.from_numpy(data["truth_" + k][cs:ce]).to(device)
                    for k in truth_keys
                }
                if has_compact_edge:
                    t_batch["edge_x"] = _build_gpu_edge_x(
                        g_edge_gpu,
                        g_eidx_gpu[cs:ce],
                        t_batch["star_x"],
                        t_batch["owner_rel"],
                    )
                else:
                    t_batch["edge_x"] = torch.from_numpy(
                        data["truth_edge_x"][cs:ce]
                    ).to(device)
                _, tv_chunk, _ = model.forward_truth_critic(t_batch)
                truth_vals_chunks.append(tv_chunk.cpu().numpy())
            truth_vals = np.concatenate(truth_vals_chunks, axis=0)

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
            for p_name in PART_NAMES:
                if "lp_" + p_name in data:
                    old_lp_parts[p_name].append(data["lp_" + p_name].astype(np.float32))

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
    if edge_tables:
        dataset["edge_static"] = torch.from_numpy(np.concatenate(edge_tables, axis=0))
        dataset["step_edge_idx"] = torch.from_numpy(
            np.concatenate(step_edge_indices, axis=0)
        )
    else:
        dataset["in_edge_x"] = torch.from_numpy(np.concatenate(legacy_in_edge, axis=0))
        dataset["truth_edge_x"] = torch.from_numpy(
            np.concatenate(legacy_truth_edge, axis=0)
        )

    for g in HEAD_GROUPS:
        dataset["old_logp_" + g] = torch.from_numpy(
            np.concatenate(old_logp_groups[g], axis=0)
        )
    if all(len(old_lp_parts[p_name]) > 0 for p_name in PART_NAMES):
        for p_name in PART_NAMES:
            dataset["old_lp_" + p_name] = torch.from_numpy(
                np.concatenate(old_lp_parts[p_name], axis=0)
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
    """Factored Per-Head PPO + Dual Distributional Critic + Aux Belief Update.

    Decouples Actor KL early-stopping from Critic updates: if any policy head
    group hits `MAX_GROUP_KL`, policy updates stop cleanly while `truth_critic`
    (and detached `fog_critic` / `belief_head` output layers) finish all epochs
    so critic accuracy never oscillates."""
    model.train()
    N = dataset["adv"].shape[0]
    n_players = dataset["in_player_x"].shape[1]

    has_compact_edge = "edge_static" in dataset
    edge_static_gpu = (
        dataset["edge_static"].to(device) if has_compact_edge else None
    )

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

    has_parts = all("old_lp_" + p_name in dataset for p_name in PART_NAMES)
    part_to_group = {
        "frontier": "spatial",
        "posture": "spatial",
        "target_player": "spatial",
        "budget": "budget",
        "knobs": "knobs",
        "research": "macro",
        "spec": "macro",
        "spec_star": "macro",
    }

    actor_stopped = False
    for _ in range(epochs):
        epoch_kls = []
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue

            b_in = {k: dataset["in_" + k][idx].to(device) for k in in_keys}
            b_truth = {k: dataset["truth_" + k][idx].to(device) for k in truth_keys}
            if has_compact_edge:
                b_eidx = dataset["step_edge_idx"][idx].to(device)
                b_in["edge_x"] = _build_gpu_edge_x(
                    edge_static_gpu, b_eidx, b_in["star_x"], b_in["owner_rel"]
                )
                b_truth["edge_x"] = _build_gpu_edge_x(
                    edge_static_gpu, b_eidx, b_truth["star_x"], b_truth["owner_rel"]
                )

            target_rank = dataset["target_rank"][idx].to(device)
            bonus_ret = dataset["bonus_ret"][idx].to(device)

            # 1. Privileged Truth Distributional Critic (disjoint critic_backbone)
            t_rank_logits, t_total_val, t_place_val = model.forward_truth_critic(b_truth)
            t_bonus_pred = t_total_val - t_place_val
            truth_val_loss = F.cross_entropy(t_rank_logits, target_rank) + F.smooth_l1_loss(
                t_bonus_pred, bonus_ret
            )

            if not actor_stopped:
                b_act = {k: dataset["act_" + k][idx].to(device) for k in act_keys}
                adv = dataset["adv"][idx].to(device)

                eval_out = model.act_or_evaluate(b_in, actions=b_act, sample=False)
                p_out = eval_out["policy_out"]

                ent_bonus = adv.new_zeros(())
                for g in HEAD_GROUPS:
                    ent_bonus = ent_bonus + ENTROPY_WEIGHTS[g] * eval_out["entropy_groups"][g].mean()

                if has_parts:
                    # Per-column independent ratio clipping + DUAL_CLIP (`ppo_train.py`)
                    total_obj = torch.zeros_like(adv)
                    grp_kl_sum = {g: 0.0 for g in HEAD_GROUPS}
                    grp_clip_num = {g: 0.0 for g in HEAD_GROUPS}
                    grp_clip_den = {g: 0.0 for g in HEAD_GROUPS}

                    for p_name in PART_NAMES:
                        new_lp = eval_out["logp_parts"][p_name]
                        old_lp = dataset["old_lp_" + p_name][idx].to(device)
                        mask_b = eval_out["part_masks"][p_name].to(new_lp.dtype)
                        log_ratio = (new_lp - old_lp).clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
                        obj_cols = _dual_clipped_objective(log_ratio, adv) * mask_b
                        total_obj = total_obj + obj_cols.sum(dim=-1)

                        with torch.no_grad():
                            ratio = log_ratio.exp()
                            kl_p = float(
                                (((ratio - 1.0) - log_ratio) * mask_b).sum().item()
                                / max(1, new_lp.shape[0])
                            )
                            c_num = float(
                                (((ratio - 1.0).abs() > CLIP_EPS).to(new_lp.dtype) * mask_b)
                                .sum()
                                .item()
                            )
                            c_den = float(mask_b.sum().item())
                            g_key = part_to_group[p_name]
                            grp_kl_sum[g_key] += kl_p
                            grp_clip_num[g_key] += c_num
                            grp_clip_den[g_key] += c_den

                    pol_loss = -total_obj.mean()
                    step_kl = sum(grp_kl_sum.values())
                    epoch_kls.append(step_kl)
                    for g in HEAD_GROUPS:
                        stats_acc[f"kl_{g}"].append(grp_kl_sum[g])
                        stats_acc[f"clip_{g}"].append(
                            grp_clip_num[g] / max(grp_clip_den[g], 1.0)
                        )
                else:
                    pol_loss = adv.new_zeros(())
                    step_kl = 0.0
                    for g in HEAD_GROUPS:
                        new_lp = eval_out["logp_groups"][g]
                        old_lp = dataset["old_logp_" + g][idx].to(device)
                        log_ratio = (new_lp - old_lp).clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO)
                        ratio = log_ratio.exp()
                        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                        clip_frac = float(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())
                        pol_loss = pol_loss - _dual_clipped_objective(log_ratio, adv).mean()
                        step_kl += approx_kl
                        stats_acc[f"kl_{g}"].append(approx_kl)
                        stats_acc[f"clip_{g}"].append(clip_frac)
                    epoch_kls.append(step_kl)

                f_rank_logits = p_out["fog_rank_logits"]
                f_bonus_pred = p_out["fog_value"] - p_out["fog_place_val"]
                fog_val_loss = F.cross_entropy(f_rank_logits, target_rank) + F.smooth_l1_loss(
                    f_bonus_pred, bonus_ret
                )

                belief_loss = compute_auxiliary_belief_loss(
                    p_out["belief_logits"],
                    dataset["aux_owner"][idx].to(device),
                    dataset["aux_vis"][idx].to(device),
                    dataset["aux_ships"][idx].to(device),
                    dataset["aux_infra"][idx].to(device),
                    n_players=n_players,
                    star_x=b_in["star_x"],
                )

                total_loss = (
                    pol_loss
                    - ent_bonus
                    + TRUTH_VALUE_COEF * truth_val_loss
                    + FOG_VALUE_COEF * fog_val_loss
                    + AUX_BELIEF_COEF * belief_loss
                )
                stats_acc["policy_loss"].append(float(pol_loss.item()))
                stats_acc["fog_val_loss"].append(float(fog_val_loss.item()))
                stats_acc["belief_loss"].append(float(belief_loss.item()))
                stats_acc["entropy"].append(float(eval_out["entropy"].mean().item()))
            else:
                # Epoch mean KL reached TARGET_KL; finish critic & detached head epochs!
                with torch.no_grad():
                    h_s_det, _, g_det = model.backbone(
                        b_in["star_x"],
                        b_in["edge_x"],
                        b_in["player_x"],
                        b_in["owner_rel"],
                        b_in["time_x"],
                    )
                f_rank_logits, f_tot_v, f_pl_v = model.fog_critic(g_det, n_players)
                fog_val_loss = F.cross_entropy(f_rank_logits, target_rank) + F.smooth_l1_loss(
                    f_tot_v - f_pl_v, bonus_ret
                )
                belief_loss = compute_auxiliary_belief_loss(
                    model.belief_head(h_s_det),
                    dataset["aux_owner"][idx].to(device),
                    dataset["aux_vis"][idx].to(device),
                    dataset["aux_ships"][idx].to(device),
                    dataset["aux_infra"][idx].to(device),
                    n_players=n_players,
                    star_x=b_in["star_x"],
                )
                total_loss = (
                    TRUTH_VALUE_COEF * truth_val_loss
                    + FOG_VALUE_COEF * fog_val_loss
                    + AUX_BELIEF_COEF * belief_loss
                )
                stats_acc["fog_val_loss"].append(float(fog_val_loss.item()))
                stats_acc["belief_loss"].append(float(belief_loss.item()))

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            stats_acc["truth_val_loss"].append(float(truth_val_loss.item()))

        if not actor_stopped and epoch_kls and float(np.mean(epoch_kls)) > TARGET_KL:
            actor_stopped = True

    model.eval()
    return {
        k: round(float(np.mean(v)), 5) if v else 0.0
        for k, v in stats_acc.items()
    }


def _run_eval_entry(
    it: int,
    ema_out: Path,
    eval_path: Path,
    eval_games: int,
    eval_workers: int,
    max_ticks: int,
    cmd_opponent: str | None = None,
):
    eval_seeds = list(range(950_000, 950_000 + eval_games))
    ev_builtin = evaluate_suite(
        eval_seeds,
        checkpoint=str(ema_out),
        workers=eval_workers,
        max_ticks=max_ticks,
    )
    ev_entry = {"iter": int(it), "vs_builtin": ev_builtin}
    if cmd_opponent and os.path.exists(cmd_opponent):
        ev_cmd = evaluate_suite(
            eval_seeds,
            checkpoint=str(ema_out),
            opponent=cmd_opponent,
            workers=eval_workers,
            max_ticks=max_ticks,
        )
        ev_entry["vs_commander"] = ev_cmd
    with open(eval_path, "a") as fp:
        fp.write(json.dumps(ev_entry) + "\n")
    print("EVAL:", json.dumps(ev_entry), flush=True)


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

    # Resume latest numbered ckpt_<iter>.pt if present
    existing_ckpts = sorted(
        [p for p in run_dir.glob("ckpt_*.pt") if re.match(r"^ckpt_\d+\.pt$", p.name)],
        key=lambda p: int(re.match(r"^ckpt_(\d+)\.pt$", p.name).group(1)),
    )
    start_iter = 0
    if existing_ckpts:
        latest = existing_ckpts[-1]
        start_iter = int(re.match(r"^ckpt_(\d+)\.pt$", latest.name).group(1))
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model = load_admiral_model(n_techs, latest, d=args.d_model, n_layers=args.n_layers).to(device)
        optimizer = make_admiral_optimizer(
            model, lr_trunk=args.lr_trunk, lr_heads=args.lr
        )
        if "optimizer" in saved:
            try:
                optimizer.load_state_dict(saved["optimizer"])
                optimizer.param_groups[0]["lr"] = args.lr_trunk
                optimizer.param_groups[-1]["lr"] = args.lr
            except ValueError:
                pass
        ema_path = run_dir / f"ema_{start_iter}.pt"
        ema_model = load_admiral_model(
            n_techs, ema_path if ema_path.exists() else latest, d=args.d_model, n_layers=args.n_layers
        )
    else:
        model = AdmiralNet(n_techs, d=args.d_model, n_layers=args.n_layers)
        if args.pretrain:
            loaded_keys = load_pretrained_backbone(model, args.pretrain)
            print(
                f"Loaded {len(loaded_keys)} pre-trained backbone/critic/belief tensors from {args.pretrain}",
                flush=True,
            )
        model = model.to(device)
        optimizer = make_admiral_optimizer(
            model, lr_trunk=args.lr_trunk, lr_heads=args.lr
        )
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

    use_async = bool(args.async_rollouts)
    use_async_eval = bool(args.async_eval or (use_async and not args.sync_eval))
    eval_workers = (
        args.eval_workers
        if args.eval_workers > 0
        else max(1, min(2, args.workers // 4))
    )

    # Clean up any interrupted synchronous rollout directory > start_iter
    if not args.keep_rollouts:
        for stale_dir in run_dir.glob("rollout_*"):
            m = re.match(r"^rollout_(\d+)$", stale_dir.name)
            if m and int(m.group(1)) > start_iter:
                shutil.rmtree(stale_dir, ignore_errors=True)

    async_dir = run_dir / "async_rollouts"
    latest_policy_path = run_dir / "latest_policy.pt"
    if use_async:
        if async_dir.exists() and not args.keep_rollouts:
            shutil.rmtree(async_dir, ignore_errors=True)
        async_dir.mkdir(exist_ok=True)
        save_admiral_checkpoint(latest_policy_path, model, None, start_iter)

    pool = Pool(max(1, args.workers)) if (args.workers > 1 or use_async) else None
    active_eval_proc: subprocess.Popen | None = None
    normal_completion = False

    # Async rollout queue state
    in_flight: list[tuple[int, any]] = []
    ready_games: deque[tuple[int, Path, int]] = deque()
    async_meta: dict[int, tuple[list[int], dict]] = {}
    game_seq = 0
    in_flight_limit = max(args.workers * 2, 4)
    n_opp = presets.PRESET["league_opponents"]

    def _top_up_async_queue():
        nonlocal game_seq
        assert pool is not None
        while len(in_flight) < in_flight_limit:
            game_seq += 1
            g_seed = args.seed + (start_iter + 1) * 100_000 + game_seq
            g_rng = np.random.default_rng(g_seed)
            l_seats = sorted(
                int(s)
                for s in g_rng.choice(
                    presets.PLAYERS, presets.SEATS, replace=False
                )
            )
            opps, ai_cfgs = league.sample_matchup(
                rng,
                l_seats,
                n_opp,
                max_commander_seats=args.max_commander_seats,
            )
            rec_file = (
                str(belief_dir / f"async_s{g_seed}.sbd")
                if args.record_every > 0 and (game_seq % args.record_every == 0)
                else None
            )
            async_meta[g_seed] = (l_seats, opps)
            job = (
                str(async_dir),
                g_seed,
                presets.SEATS,
                str(latest_policy_path),
                args.max_ticks,
                l_seats,
                opps,
                rec_file,
                ai_cfgs,
                True,
            )
            fut = pool.apply_async(_async_rollout_worker, (job,))
            in_flight.append((g_seed, fut))

    try:
        if use_async:
            _top_up_async_queue()

        for it in range(start_iter + 1, start_iter + args.iters + 1):
            t0 = time.time()
            places_iter: list[int] = []
            ticks_iter: list[int] = []
            staleness_iter: list[int] = []
            dropped_stale = 0

            if use_async:
                _top_up_async_queue()
                batch_npz: list[Path] = []
                while len(batch_npz) < args.games_per_iter:
                    still_running = []
                    newly_done = False
                    for g_seed, fut in in_flight:
                        if fut.ready():
                            r_seed, npz_str, pol_iter, _ = fut.get()
                            ready_games.append((r_seed, Path(npz_str), pol_iter))
                            newly_done = True
                        else:
                            still_running.append((g_seed, fut))
                    in_flight = still_running
                    if newly_done:
                        _top_up_async_queue()

                    while ready_games and len(batch_npz) < args.games_per_iter:
                        g_seed, fpath, pol_iter = ready_games.popleft()
                        if not fpath.exists():
                            continue
                        d = np.load(fpath)
                        meta = async_meta.pop(g_seed, None)
                        if meta is not None:
                            l_seats, opps = meta
                            league.update_outcomes(opps, l_seats, d["place_all"])

                        staleness = max(0, (it - 1) - int(pol_iter))
                        if staleness <= args.max_staleness:
                            batch_npz.append(fpath)
                            places_iter.extend(d["place"].tolist())
                            ticks_iter.append(int(d["ticks"]))
                            staleness_iter.append(staleness)
                        else:
                            dropped_stale += 1
                            if not args.keep_rollouts:
                                fpath.unlink(missing_ok=True)

                    if len(batch_npz) < args.games_per_iter:
                        time.sleep(0.03)

                npz_files = batch_npz
                roll_dir = None
            else:
                roll_dir = run_dir / f"rollout_{it}"
                roll_dir.mkdir(exist_ok=True)
                cur_ckpt = run_dir / f"ckpt_{it - 1}.pt"
                if not cur_ckpt.exists():
                    save_admiral_checkpoint(cur_ckpt, model, optimizer, it - 1)

                jobs = []
                job_meta = {}
                base_seed = args.seed + it * 10_000

                for g_idx in range(args.games_per_iter):
                    g_seed = base_seed + g_idx
                    g_rng = np.random.default_rng(g_seed)
                    l_seats = sorted(
                        int(s)
                        for s in g_rng.choice(
                            presets.PLAYERS, presets.SEATS, replace=False
                        )
                    )
                    opps, ai_cfgs = league.sample_matchup(
                        rng,
                        l_seats,
                        n_opp,
                        max_commander_seats=args.max_commander_seats,
                    )
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
                            l_seats,
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
                for f in npz_files:
                    d = np.load(f)
                    g_seed = int(d["seed"])
                    places_iter.extend(d["place"].tolist())
                    ticks_iter.append(int(d["ticks"]))
                    staleness_iter.append(0)
                    if g_seed in job_meta:
                        l_seats, opps = job_meta[g_seed]
                        league.update_outcomes(opps, l_seats, d["place_all"])

            # GPU PPO + Critic + Conservation-Belief update
            # (In `--async-rollouts` mode, CPU workers continue simulating
            # `in_flight` games uninterrupted while this runs!)
            t_gpu = time.time()
            dataset = build_dataset_from_rollouts(npz_files, model, device)
            upd_stats = ppo_update(
                model,
                optimizer,
                dataset,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            gpu_sec = time.time() - t_gpu
            update_ema(ema_model, model, decay=EMA_DECAY)

            if use_async:
                # Immediately publish new policy weights for the next worker game!
                save_admiral_checkpoint(latest_policy_path, model, None, it)

            ckpt_out = run_dir / f"ckpt_{it}.pt"
            ema_out = run_dir / f"ema_{it}.pt"
            save_admiral_checkpoint(ckpt_out, model, optimizer, it)
            save_admiral_checkpoint(ema_out, ema_model, None, it)

            if it % args.snapshot_every == 0:
                league.register_member(f"admiral_ema_{it}", ema_out, kind="admiral", step=it)
            league.save()

            if not args.keep_rollouts:
                if use_async:
                    for f in npz_files:
                        f.unlink(missing_ok=True)
                elif roll_dir is not None:
                    shutil.rmtree(roll_dir, ignore_errors=True)

                prune_it = it - 2
                is_milestone = (
                    prune_it <= 0
                    or (prune_it % args.snapshot_every == 0)
                    or (args.eval_every > 0 and prune_it % args.eval_every == 0)
                )
                if not is_milestone:
                    old_ckpt = run_dir / f"ckpt_{prune_it}.pt"
                    if old_ckpt.exists():
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
                "gpu_sec": round(gpu_sec, 2),
                **upd_stats,
            }
            if use_async:
                entry["mean_staleness"] = round(float(np.mean(staleness_iter)), 3)
                if dropped_stale > 0:
                    entry["dropped_stale"] = dropped_stale

            with open(log_path, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)

            if args.eval_every > 0 and it % args.eval_every == 0:
                cmd_members = [
                    v["path"]
                    for v in league.members.values()
                    if v["kind"] == "commander" and os.path.exists(v["path"])
                ]
                cmd_opp = cmd_members[-1] if cmd_members else None

                if use_async_eval:
                    if active_eval_proc is not None and active_eval_proc.poll() is None:
                        active_eval_proc.wait()
                    eval_cmd = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--run-dir",
                        str(run_dir),
                        "--internal-eval-ckpt",
                        str(ema_out),
                        "--internal-eval-iter",
                        str(it),
                        "--internal-eval-out",
                        str(eval_path),
                        "--eval-games",
                        str(args.eval_games),
                        "--eval-workers",
                        str(eval_workers),
                        "--max-ticks",
                        str(args.max_ticks),
                    ]
                    if cmd_opp:
                        eval_cmd.extend(["--internal-eval-opponent", cmd_opp])
                    active_eval_proc = subprocess.Popen(eval_cmd, cwd=os.getcwd())
                else:
                    _run_eval_entry(
                        it=it,
                        ema_out=ema_out,
                        eval_path=eval_path,
                        eval_games=args.eval_games,
                        eval_workers=args.workers,
                        max_ticks=args.max_ticks,
                        cmd_opponent=cmd_opp,
                    )

        normal_completion = True
    finally:
        if pool is not None:
            if use_async:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        if active_eval_proc is not None and active_eval_proc.poll() is None:
            if normal_completion:
                active_eval_proc.wait()
            else:
                active_eval_proc.terminate()


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
    p.add_argument(
        "--max-commander-seats",
        type=int,
        default=2,
        help="Max Commander cross-play seats per game (auto-capped to 1 while win-rate <25%%).",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-ticks", type=int, default=3000)
    p.add_argument(
        "--pretrain",
        default=None,
        help="Optional pre-trained Admiral belief/critic .pt checkpoint (from admiral_pretrain.py) when starting iter 0.",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=LR_HEADS, help="Learning rate for policy heads & critic.")
    p.add_argument("--lr-trunk", type=float, default=LR_TRUNK, help="Learning rate for pre-trained fog Graph Transformer.")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--record-every", type=int, default=4, help="Save .sbd belief file every N games (0=off).")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=16)
    p.add_argument("--eval-workers", type=int, default=0, help="Workers for async eval (default: auto 1-2).")
    p.add_argument(
        "--async-rollouts",
        action="store_true",
        help="Continuously run games without iteration barriers, overlapping rollouts with GPU PPO.",
    )
    p.add_argument(
        "--async-eval",
        action="store_true",
        help="Run periodic evaluations in a non-blocking background subprocess (auto-on with --async-rollouts).",
    )
    p.add_argument(
        "--sync-eval",
        action="store_true",
        help="Force blocking synchronous evaluation even when --async-rollouts is enabled.",
    )
    p.add_argument(
        "--max-staleness",
        type=int,
        default=3,
        help="Maximum policy lag (iterations) allowed for async straggler games.",
    )
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--keep-rollouts", action="store_true")

    # Internal flags used when spawning non-blocking background evaluation
    p.add_argument("--internal-eval-ckpt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--internal-eval-iter", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--internal-eval-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--internal-eval-opponent", default=None, help=argparse.SUPPRESS)

    args = p.parse_args()
    if args.internal_eval_ckpt:
        _run_eval_entry(
            it=args.internal_eval_iter,
            ema_out=Path(args.internal_eval_ckpt),
            eval_path=Path(args.internal_eval_out),
            eval_games=args.eval_games,
            eval_workers=max(1, args.eval_workers),
            max_ticks=args.max_ticks,
            cmd_opponent=args.internal_eval_opponent,
        )
        return

    train(args)


if __name__ == "__main__":
    main()
