"""Supervised Multi-Task Representation Pre-Trainer (`admiral_pretrain.py`).

Pre-trains `AdmiralNet`'s dual Heterogeneous Graph Transformers (`backbone` for
fog-of-war observations and `critic_backbone` for privileged full-galaxy truth),
`belief_head` (4-channel conservation residuals), and dual distributional rank
critics (`fog_critic`, `truth_critic`) offline from `SBD1` `.sbd` game recordings
before launching `admiral_train.py --pretrain <admiral_belief.pt>`.

Why this matters:
- Training a 4-layer Star<->Player Graph Transformer from random init directly
  under noisy 10-player PPO gradients causes representation collapse (flat
  `belief_loss` and state-independent policy outputs).
- Offline `.sbd` pre-training gives every star & player token dense supervised
  gradients on:
    1. Hidden garrison & infrastructure residuals (`belief_kl` on top of `BeliefTracker`),
    2. True star garrison, 1-cycle enemy reach threat, and 1-cycle incoming fleets,
    3. Hidden opponent treasury, specialist tokens, and active research topic,
    4. Final 10-player game placement from both fog (`fog_critic`) and truth (`truth_critic`),
    5. 2-cycle future star ownership (`truth_backbone` spatial dynamics).
- Policy heads (`frontier_head`, `budget_head`, `research_*`, etc.) remain
  untouched at their clean zero-initialized iter-0 priors.

Example usage:
    python3 sim/admiral/python/admiral_pretrain.py \
        --sbd-dirs data/admiral2/belief data/admiral1/belief data/league1/belief \
        --max-games 600 --samples-per-game 24 --epochs 10 --workers 8 \
        --out data/admiral_belief.pt
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_UPSTREAM_PY = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "python"))
for _p in (_THIS_DIR, _UPSTREAM_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import belief_data as bd  # noqa: E402
import presets  # noqa: E402
from admiral_net import VISIBLE, AdmiralNet, build_admiral_inputs  # noqa: E402
from admiral_train import (  # noqa: E402
    _build_gpu_edge_x,
    compute_auxiliary_belief_loss,
    save_admiral_checkpoint,
)
from belief_tracker import BeliefTracker  # noqa: E402


def _extract_natural_resources(stars_header: list[dict]) -> np.ndarray:
    """Return `[S, 3]` natural resources `(economy, industry, science)` from `.sbd` header."""
    out = np.zeros((len(stars_header), 3), dtype=np.float32)
    for i, s in enumerate(stars_header):
        nr = s.get("naturalResources") or {}
        if not isinstance(nr, dict):
            nr = {"economy": nr, "industry": nr, "science": nr}
        out[i, 0] = float(nr.get("economy") or 0.0)
        out[i, 1] = float(nr.get("industry") or 0.0)
        out[i, 2] = float(nr.get("science") or 0.0)
    return out


def _reconstruct_own_facts(
    g: bd.Game, t_idx: int, seat: int, nr_arr: np.ndarray
) -> dict:
    """Reconstruct exact Solaris marginal `upgradeCosts` & research progress for
    `seat` at recorded tick index `t_idx` (computed strictly on `owner == seat`)."""
    S = g.S
    owner = g.owner[t_idx]
    own_mask = owner == seat
    dead = (nr_arr[:, 0] <= 0.0) & (nr_arr[:, 1] <= 0.0) & (nr_arr[:, 2] <= 0.0)
    valid_own = own_mask & (~dead)

    terra_idx = g.techs.index("terraforming") if "terraforming" in g.techs else -1
    terra_lvl = float(g.levels[t_idx, seat, terra_idx]) if terra_idx >= 0 else 1.0
    tr = np.floor(nr_arr + 5.0 * terra_lvl)
    cur_inf = g.infra[t_idx].astype(np.float32)

    eco_costs = np.full(S, -1.0, dtype=np.float32)
    ind_costs = np.full(S, -1.0, dtype=np.float32)
    sci_costs = np.full(S, -1.0, dtype=np.float32)
    wg_costs = np.full(S, -1.0, dtype=np.float32)

    m_eco = valid_own & (tr[:, 0] > 0.0)
    if m_eco.any():
        eco_costs[m_eco] = np.maximum(
            1.0, np.floor((5.0 * (cur_inf[m_eco, 0] + 1.0)) / (tr[m_eco, 0] / 100.0))
        )
    m_ind = valid_own & (tr[:, 1] > 0.0)
    if m_ind.any():
        ind_costs[m_ind] = np.maximum(
            1.0, np.floor((10.0 * (cur_inf[m_ind, 1] + 1.0)) / (tr[m_ind, 1] / 100.0))
        )
    m_sci = valid_own & (tr[:, 2] > 0.0)
    if m_sci.any():
        sci_costs[m_sci] = np.maximum(
            1.0, np.floor((40.0 * (cur_inf[m_sci, 2] + 1.0)) / (tr[m_sci, 2] / 100.0))
        )

    avg_tr = np.floor((tr[:, 0] + tr[:, 1] + tr[:, 2]) / 3.0)
    m_wg = valid_own & (g.warp[t_idx] <= 0) & (avg_tr > 0.0)
    if m_wg.any():
        wg_costs[m_wg] = np.maximum(1.0, np.floor(100.0 / (avg_tr[m_wg] / 100.0)))

    credits = float(g.players[t_idx, seat, g.pf["credits"]])
    spec_budget = float(g.players[t_idx, seat, g.pf["creditsSpecialists"]])
    res_now = int(g.players[t_idx, seat, g.pf["researchingNow"]])
    res_prog = g.progress[t_idx, seat].astype(np.float32)
    res_req = np.array(
        [
            float(g.levels[t_idx, seat, ti]) * 100.0
            if t_name in g.researchable
            else 0.0
            for ti, t_name in enumerate(g.techs)
        ],
        dtype=np.float32,
    )

    return {
        "credits": credits,
        "specialistBudget": spec_budget,
        "researchingNow": res_now,
        "researchProgress": res_prog,
        "researchRequired": res_req,
        "upgradeCosts": {
            "economy": eco_costs,
            "industry": ind_costs,
            "science": sci_costs,
            "warpGate": wg_costs,
        },
        "affordableStarSpecialists": [],
    }


def _build_truth_frame_and_facts(g: bd.Game, t_idx: int) -> tuple[dict, dict]:
    """Construct privileged omniscient `truth_frame` & `truth_facts` at `t_idx`."""
    tick = int(g.ticks[t_idx])
    S = g.S
    rows = g.carriers_at(t_idx)
    all_car = g._carrier_dict(rows)

    extras = np.stack(
        [
            (g.warp[t_idx] > 0).astype(np.float32),
            (g.specialist[t_idx] >= 0).astype(np.float32),
        ],
        axis=1,
    )
    truth_frame = {
        "tick": tick,
        "owner": g.owner[t_idx],
        "visibility": np.full(S, VISIBLE, dtype=np.uint8),
        "seen_tick": np.full(S, tick, dtype=np.int32),
        "mem_ships": g.ships[t_idx].astype(np.float32),
        "mem_infra": g.infra[t_idx].astype(np.float32),
        "mem_extras": extras,
        "carriers": all_car,
        "levels": g.levels[t_idx],
        "stats": g.public_stats(t_idx),
        "defeated": g.players[t_idx, :, g.pf["defeated"]],
    }
    truth_facts = {
        "playerCredits": g.players[t_idx, :, g.pf["credits"]].astype(np.float32),
        "playerSpecTokens": g.players[
            t_idx, :, g.pf["creditsSpecialists"]
        ].astype(np.float32),
        "playerResearchingNow": g.players[
            t_idx, :, g.pf["researchingNow"]
        ].astype(np.int32),
        "playerResearchProgress": g.progress[t_idx].astype(np.float32),
    }
    return truth_frame, truth_facts


def _compute_final_ranks(g: bd.Game) -> np.ndarray:
    """Compute zero-indexed final placements `0..P-1` at `g.T - 1` matching `admiral_rollout.placements`."""
    t_end = g.T - 1
    winner = g.header.get("winner", None)
    end_stats = g.public_stats(t_end)
    end_def = g.players[t_end, :, g.pf["defeated"]] > 0
    if winner is None:
        win_thresh = math.ceil(g.S * 0.5)
        for q in range(g.P):
            if int(end_stats[q, 0]) >= win_thresh and not end_def[q]:
                winner = q
                break

    order = sorted(
        range(g.P),
        key=lambda q: (
            winner != q,
            bool(end_def[q]),
            -float(end_stats[q, 0]),
            -float(end_stats[q, 1]),
        ),
    )
    ranks = np.zeros(g.P, dtype=np.int64)
    for rank_idx, seat in enumerate(order):
        ranks[seat] = rank_idx
    return ranks


def _extract_game_worker(job: tuple) -> dict | None:
    """Worker: extract multi-task fog + truth pre-training samples from one `.sbd` file."""
    (
        game_id,
        sbd_path,
        samples_per_game,
        expected_players,
        expected_stars,
        seed,
    ) = job
    try:
        g = bd.Game(sbd_path)
    except Exception:
        return None

    if g.P != expected_players or g.S != expected_stars or g.T < 2:
        return None

    rng = np.random.default_rng(int(seed) + int(game_id) * 1009)
    nr_arr = _extract_natural_resources(g.header["stars"])
    final_ranks = _compute_final_ranks(g)

    start_t = min(g.cycle, max(1, g.T // 4))
    cand_ticks = np.arange(start_t, g.T, dtype=np.int32)
    if len(cand_ticks) == 0:
        cand_ticks = np.arange(0, g.T, dtype=np.int32)

    # Pick up to 4 seats that survive past start_t so we compute `g.memory(seat)` at most 4 times
    alive_seats = [
        p
        for p in range(g.P)
        if g.players[start_t, p, g.pf["defeated"]] == 0
        and (g.owner[start_t] == p).any()
    ]
    if not alive_seats:
        alive_seats = list(range(g.P))

    n_seats_pick = min(len(alive_seats), 4, max(1, samples_per_game))
    chosen_seats = [
        int(x) for x in rng.choice(alive_seats, size=n_seats_pick, replace=False)
    ]
    per_seat_quota = max(1, int(math.ceil(samples_per_game / float(n_seats_pick))))

    fog_star_list = []
    fog_player_list = []
    owner_rel_list = []
    time_x_list = []
    truth_star_list = []
    truth_player_list = []

    aux_owner_list = []
    aux_vis_list = []
    aux_ships_list = []
    aux_infra_list = []
    star_threat_list = []
    opp_cred_list = []
    opp_res_list = []
    opp_alive_list = []
    final_rank_list = []
    future_own_list = []

    edge_map: dict[int, int] = {}
    edge_slices: list[np.ndarray] = []
    step_edge_idx: list[int] = []

    collected = 0
    for seat in chosen_seats:
        if collected >= samples_per_game:
            break
        valid_t = [
            int(t)
            for t in cand_ticks
            if g.players[t, seat, g.pf["defeated"]] == 0
            and (g.owner[t] == seat).any()
        ]
        if not valid_t:
            continue
        k_take = min(len(valid_t), per_seat_quota, samples_per_game - collected)
        if len(valid_t) <= k_take:
            picked_t = sorted(valid_t)
        else:
            # Stratified ticks across early, mid, and late war!
            lin_idx = np.linspace(0, len(valid_t) - 1, k_take)
            jitter = rng.integers(-2, 3, size=k_take)
            idx_sel = np.clip(np.round(lin_idx).astype(int) + jitter, 0, len(valid_t) - 1)
            picked_t = sorted(set(valid_t[i] for i in idx_sel))

        mem = g.memory(seat)
        for t_idx in picked_t:
            fog_tracker = BeliefTracker(g, seat)
            truth_tracker = BeliefTracker(g, seat)
            for prev_t in sorted({max(0, t_idx - g.cycle), max(0, t_idx - 4)}):
                if prev_t < t_idx:
                    fog_tracker.update(g.view(prev_t, seat, mem))
                    tf_prev, _ = _build_truth_frame_and_facts(g, prev_t)
                    truth_tracker.update(tf_prev)

            fog_view = g.view(t_idx, seat, mem)
            own_facts = _reconstruct_own_facts(g, t_idx, seat, nr_arr)
            fog_in = build_admiral_inputs(
                g, fog_view, seat, own_facts=own_facts, tracker=fog_tracker
            )

            truth_frame, truth_facts = _build_truth_frame_and_facts(g, t_idx)
            truth_in = build_admiral_inputs(
                g,
                truth_frame,
                seat,
                own_facts=own_facts,
                tracker=truth_tracker,
                truth_facts=truth_facts,
            )

            own_hs_lvl = int(g.hyperspace(g.levels[t_idx], seat))
            if own_hs_lvl not in edge_map:
                edge_map[own_hs_lvl] = len(edge_slices)
                edge_slices.append(fog_in["edge_x"][:, :, :3].astype(np.float16))
            step_edge_idx.append(edge_map[own_hs_lvl])

            # Multi-task targets
            true_ships = g.ships[t_idx].astype(np.float32)
            all_car = g._carrier_dict(g.carriers_at(t_idx))
            threat_1c = g.threat(
                seat,
                g.owner[t_idx],
                g.levels[t_idx],
                true_ships,
                all_car,
                all_car["ships"],
                1,
            )
            inc_1c = g._incoming(t_idx, seat)
            threat_y = (
                np.log1p(
                    np.maximum(
                        0.0, np.stack([true_ships, threat_1c, inc_1c], axis=1)
                    )
                )
                / 5.0
            ).astype(np.float32)

            p_order = [seat] + [q for q in range(g.P) if q != seat]
            pl_row = g.players[t_idx]
            cred_y = np.stack(
                [
                    np.log1p(np.maximum(0.0, pl_row[p_order, g.pf["credits"]].astype(np.float32)))
                    / 8.0,
                    np.log1p(
                        np.maximum(
                            0.0,
                            pl_row[p_order, g.pf["creditsSpecialists"]].astype(np.float32),
                        )
                    )
                    / 4.0,
                ],
                axis=1,
            ).astype(np.float32)
            res_y = pl_row[p_order, g.pf["researchingNow"]].astype(np.int64)
            alive_y = (pl_row[p_order, g.pf["defeated"]] == 0) & (
                truth_frame["stats"][p_order, 0] > 0
            )

            t_fut = min(g.T - 1, t_idx + 2 * g.cycle)
            fut_own = (g.owner[t_fut] == seat).astype(np.float32)

            fog_star_list.append(fog_in["star_x"])
            fog_player_list.append(fog_in["player_x"])
            owner_rel_list.append(fog_in["owner_rel"].astype(np.int16))
            time_x_list.append(fog_in["time_x"])
            truth_star_list.append(truth_in["star_x"])
            truth_player_list.append(truth_in["player_x"])

            aux_owner_list.append(g.owner[t_idx].astype(np.int16))
            aux_vis_list.append(fog_view["visibility"].astype(np.uint8))
            aux_ships_list.append(true_ships)
            aux_infra_list.append(g.infra[t_idx].astype(np.float32))
            star_threat_list.append(threat_y)
            opp_cred_list.append(cred_y)
            opp_res_list.append(res_y)
            opp_alive_list.append(alive_y)
            final_rank_list.append(int(final_ranks[seat]))
            future_own_list.append(fut_own)
            collected += 1

    if collected == 0:
        return None

    return {
        "game_id": int(game_id),
        "n_techs": len(g.techs),
        "edge_static": np.stack(edge_slices, axis=0),
        "step_edge_idx": np.asarray(step_edge_idx, dtype=np.int32),
        "fog_star_x": np.stack(fog_star_list, axis=0),
        "fog_player_x": np.stack(fog_player_list, axis=0),
        "owner_rel": np.stack(owner_rel_list, axis=0),
        "time_x": np.stack(time_x_list, axis=0),
        "truth_star_x": np.stack(truth_star_list, axis=0),
        "truth_player_x": np.stack(truth_player_list, axis=0),
        "aux_owner": np.stack(aux_owner_list, axis=0),
        "aux_vis": np.stack(aux_vis_list, axis=0),
        "aux_ships": np.stack(aux_ships_list, axis=0),
        "aux_infra": np.stack(aux_infra_list, axis=0),
        "star_threat_y": np.stack(star_threat_list, axis=0),
        "opp_cred_y": np.stack(opp_cred_list, axis=0),
        "opp_res_y": np.stack(opp_res_list, axis=0),
        "opp_alive": np.stack(opp_alive_list, axis=0),
        "final_rank": np.asarray(final_rank_list, dtype=np.int64),
        "future_own_y": np.stack(future_own_list, axis=0),
    }


def discover_sbd_files(
    sbd_dirs: list[str | Path], max_games: int, seed: int = 42
) -> list[Path]:
    """Collect `.sbd` files balanced across input directories so both `Admiral`
    cross-play recordings and `Commander` league recordings are well-represented."""
    rng = np.random.default_rng(seed)
    per_dir_lists: list[list[Path]] = []
    for d in sbd_dirs:
        p = Path(d)
        if p.is_file() and p.suffix == ".sbd":
            per_dir_lists.append([p])
        elif p.is_dir():
            found = sorted(p.glob("*.sbd"))
            if found:
                rng.shuffle(found)
                per_dir_lists.append(found)

    if not per_dir_lists:
        return []

    if max_games <= 0:
        return [f for sub in per_dir_lists for f in sub]

    # Round-robin interleave across directories so no huge directory drowns smaller ones
    selected: list[Path] = []
    cursors = [0] * len(per_dir_lists)
    while len(selected) < max_games:
        progressed = False
        for di, sub in enumerate(per_dir_lists):
            if cursors[di] < len(sub):
                selected.append(sub[cursors[di]])
                cursors[di] += 1
                progressed = True
                if len(selected) >= max_games:
                    break
        if not progressed:
            break
    return selected


def _compute_batch_losses(
    model: AdmiralNet,
    ds: dict[str, torch.Tensor],
    edge_static_gpu: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    fog_star_x = ds["fog_star_x"][idx].to(device)
    fog_player_x = ds["fog_player_x"][idx].to(device)
    owner_rel = ds["owner_rel"][idx].to(device, dtype=torch.long)
    time_x = ds["time_x"][idx].to(device)
    truth_star_x = ds["truth_star_x"][idx].to(device)
    truth_player_x = ds["truth_player_x"][idx].to(device)
    e_idx = ds["step_edge_idx"][idx].to(device, dtype=torch.long)

    fog_edge_x = _build_gpu_edge_x(edge_static_gpu, e_idx, fog_star_x, owner_rel)
    truth_edge_x = _build_gpu_edge_x(edge_static_gpu, e_idx, truth_star_x, owner_rel)

    out = model.forward_pretrain(
        {
            "star_x": fog_star_x,
            "edge_x": fog_edge_x,
            "player_x": fog_player_x,
            "owner_rel": owner_rel,
            "time_x": time_x,
        },
        {
            "star_x": truth_star_x,
            "edge_x": truth_edge_x,
            "player_x": truth_player_x,
            "owner_rel": owner_rel,
            "time_x": time_x,
        },
    )

    P = fog_player_x.shape[1]
    T = model.n_techs
    aux_owner = ds["aux_owner"][idx].to(device, dtype=torch.long)
    aux_vis = ds["aux_vis"][idx].to(device)
    aux_ships = ds["aux_ships"][idx].to(device)
    aux_infra = ds["aux_infra"][idx].to(device)

    # 1. 4-channel conservation residual KL loss
    belief_kl = compute_auxiliary_belief_loss(
        out["belief_logits"],
        aux_owner,
        aux_vis,
        aux_ships,
        aux_infra,
        n_players=P,
        star_x=fog_star_x,
    )

    # 2. Star tactical threat & hidden garrison regression (upweight fog-hidden stars 2x)
    star_threat_y = ds["star_threat_y"][idx].to(device)
    elem_threat = F.smooth_l1_loss(out["star_threat_pred"], star_threat_y, reduction="none")
    fog_w = torch.where(aux_vis != VISIBLE, 2.0, 0.5).unsqueeze(-1)
    threat_l1 = (elem_threat * fog_w).sum() / fog_w.sum().clamp(min=1.0) / 3.0

    # 3. Opponent hidden credits, specialist tokens, and active research topic
    opp_cred_y = ds["opp_cred_y"][idx].to(device)[:, 1:]
    opp_res_y = ds["opp_res_y"][idx].to(device, dtype=torch.long)[:, 1:]
    opp_alive = ds["opp_alive"][idx].to(device, dtype=torch.bool)[:, 1:]

    opp_pred = out["opp_state_pred"][:, 1:]
    if opp_alive.any():
        cred_diff = F.smooth_l1_loss(opp_pred[:, :, :2], opp_cred_y, reduction="none").mean(dim=-1)
        opp_cred_l1 = cred_diff[opp_alive].mean()
    else:
        opp_cred_l1 = belief_kl.new_zeros(())

    valid_res = opp_alive & (opp_res_y >= 0) & (opp_res_y < T)
    if valid_res.any():
        res_logits_flat = opp_pred[:, :, 2:][valid_res]
        res_targets_flat = opp_res_y[valid_res]
        opp_res_ce = F.cross_entropy(res_logits_flat, res_targets_flat)
        opp_res_acc = float((res_logits_flat.argmax(dim=-1) == res_targets_flat).float().mean().item())
    else:
        opp_res_ce = belief_kl.new_zeros(())
        opp_res_acc = 0.0

    # 4. Fog & Truth Distributional Rank Critics
    final_rank = ds["final_rank"][idx].to(device, dtype=torch.long)
    fog_rank_ce = F.cross_entropy(out["fog_rank_logits"], final_rank)
    truth_rank_ce = F.cross_entropy(out["truth_rank_logits"], final_rank)

    # 5. Privileged 2-cycle star control projection
    future_own_y = ds["future_own_y"][idx].to(device)
    expand_bce = F.binary_cross_entropy_with_logits(out["truth_expand_logits"], future_own_y)

    total_loss = (
        1.00 * belief_kl
        + 1.50 * threat_l1
        + 0.50 * opp_cred_l1
        + 0.35 * opp_res_ce
        + 0.60 * fog_rank_ce
        + 0.60 * truth_rank_ce
        + 0.40 * expand_bce
    )
    metrics = {
        "loss": float(total_loss.item()),
        "belief_kl": float(belief_kl.item()),
        "threat_l1": float(threat_l1.item()),
        "opp_cred_l1": float(opp_cred_l1.item()),
        "opp_res_ce": float(opp_res_ce.item()),
        "opp_res_acc": opp_res_acc,
        "fog_rank_ce": float(fog_rank_ce.item()),
        "truth_rank_ce": float(truth_rank_ce.item()),
        "expand_bce": float(expand_bce.item()),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_pretrain(
    model: AdmiralNet,
    ds: dict[str, torch.Tensor],
    edge_static_gpu: torch.Tensor,
    val_idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    model.eval()
    acc: dict[str, list[float]] = {}
    weights: list[int] = []
    for start in range(0, len(val_idx), batch_size):
        b_np = val_idx[start : start + batch_size]
        idx_t = torch.from_numpy(b_np).long()
        _, m = _compute_batch_losses(model, ds, edge_static_gpu, idx_t, device)
        weights.append(len(b_np))
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    w_arr = np.asarray(weights, dtype=np.float64)
    w_sum = max(1.0, float(w_arr.sum()))
    return {
        k: round(float(np.sum(np.asarray(vals) * w_arr) / w_sum), 5)
        for k, vals in acc.items()
    }


def pretrain_admiral_from_sbd(
    sbd_dirs: list[str | Path],
    out_path: str | Path,
    max_games: int = 500,
    samples_per_game: int = 24,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.10,
    workers: int = 8,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    expected_players: int = presets.PLAYERS,
    expected_stars: int = 200,
    seed: int = 42,
) -> dict:
    """Extract multi-task samples from `.sbd` games, pre-train `AdmiralNet` on GPU/CPU,
    save checkpoint to `out_path`, and return baseline vs best validation metrics."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)

    files = discover_sbd_files(sbd_dirs, max_games=max_games, seed=seed)
    if not files:
        raise ValueError(f"No .sbd files found in {sbd_dirs}")

    t0 = time.time()
    jobs = [
        (i, str(fp), int(samples_per_game), int(expected_players), int(expected_stars), int(seed))
        for i, fp in enumerate(files)
    ]
    eff_workers = min(max(1, int(workers)), len(jobs))
    if eff_workers > 1:
        with Pool(processes=eff_workers) as pool:
            raw_chunks = pool.map(_extract_game_worker, jobs)
    else:
        raw_chunks = [_extract_game_worker(j) for j in jobs]

    chunks = [c for c in raw_chunks if c is not None]
    if not chunks:
        raise RuntimeError(
            f"No valid {expected_players}p/{expected_stars}-star .sbd games extracted from {len(files)} files."
        )

    n_techs = int(chunks[0]["n_techs"])
    edge_tables = []
    step_edge_indices = []
    game_ids = []
    edge_offset = 0
    cat_keys = (
        "fog_star_x",
        "fog_player_x",
        "owner_rel",
        "time_x",
        "truth_star_x",
        "truth_player_x",
        "aux_owner",
        "aux_vis",
        "aux_ships",
        "aux_infra",
        "star_threat_y",
        "opp_cred_y",
        "opp_res_y",
        "opp_alive",
        "final_rank",
        "future_own_y",
    )
    lists_by_key: dict[str, list[np.ndarray]] = {k: [] for k in cat_keys}

    for c in chunks:
        n_s = len(c["final_rank"])
        edge_tables.append(c["edge_static"])
        step_edge_indices.append(c["step_edge_idx"] + edge_offset)
        edge_offset += len(c["edge_static"])
        game_ids.append(np.full(n_s, c["game_id"], dtype=np.int32))
        for k in cat_keys:
            lists_by_key[k].append(c[k])

    ds: dict[str, torch.Tensor] = {
        k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in lists_by_key.items()
    }
    ds["step_edge_idx"] = torch.from_numpy(np.concatenate(step_edge_indices, axis=0))
    edge_static = torch.from_numpy(np.concatenate(edge_tables, axis=0))
    game_id_arr = np.concatenate(game_ids, axis=0)

    total_samples = len(game_id_arr)
    unique_games = np.unique(game_id_arr)
    rng = np.random.default_rng(seed)

    if len(unique_games) >= 2:
        n_val_games = max(1, int(round(len(unique_games) * val_frac)))
        val_games = set(
            rng.choice(unique_games, size=n_val_games, replace=False).tolist()
        )
        is_val = np.isin(game_id_arr, list(val_games))
        train_idx = np.flatnonzero(~is_val)
        val_idx = np.flatnonzero(is_val)
    else:
        perm = rng.permutation(total_samples)
        n_val = max(1, total_samples // 5)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:] if total_samples > n_val else perm

    extract_sec = time.time() - t0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_static_gpu = edge_static.to(device)

    print(
        f"[admiral_pretrain] extracted {total_samples} samples from {len(chunks)} games "
        f"(train={len(train_idx)}, val={len(val_idx)}, edge_slices={len(edge_static)}) "
        f"in {extract_sec:.1f}s | device={device}",
        flush=True,
    )

    model = AdmiralNet(
        n_techs=n_techs, d=d_model, n_layers=n_layers, n_heads=n_heads
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = max(1, epochs * max(1, math.ceil(len(train_idx) / float(batch_size))))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.05
    )

    val_init = evaluate_pretrain(
        model, ds, edge_static_gpu, val_idx, device, batch_size=batch_size
    )
    print(
        f"[epoch 00 baseline] val_loss={val_init['loss']:.4f} "
        f"belief_kl={val_init['belief_kl']:.4f} threat_l1={val_init['threat_l1']:.4f} "
        f"fog_rank_ce={val_init['fog_rank_ce']:.4f} truth_rank_ce={val_init['truth_rank_ce']:.4f} "
        f"opp_res_acc={val_init['opp_res_acc']:.3f}",
        flush=True,
    )

    best_val_loss = float("inf")
    best_val_metrics = val_init
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        perm = rng.permutation(train_idx)
        tr_losses = []
        for start in range(0, len(perm), batch_size):
            b_np = perm[start : start + batch_size]
            idx_t = torch.from_numpy(b_np).long()
            loss, m = _compute_batch_losses(model, ds, edge_static_gpu, idx_t, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_losses.append(m["loss"])

        val_m = evaluate_pretrain(
            model, ds, edge_static_gpu, val_idx, device, batch_size=batch_size
        )
        improved = val_m["loss"] <= best_val_loss
        if improved:
            best_val_loss = val_m["loss"]
            best_val_metrics = val_m
            save_admiral_checkpoint(out_path, model, None, 0)

        print(
            f"[epoch {ep:02d}/{epochs:02d}] train_loss={float(np.mean(tr_losses)):.4f} "
            f"val_loss={val_m['loss']:.4f} belief_kl={val_m['belief_kl']:.4f} "
            f"threat_l1={val_m['threat_l1']:.4f} fog_rank_ce={val_m['fog_rank_ce']:.4f} "
            f"truth_rank_ce={val_m['truth_rank_ce']:.4f} opp_res_acc={val_m['opp_res_acc']:.3f} "
            f"({time.time() - ep_t0:.1f}s){' *' if improved else ''}",
            flush=True,
        )

    if not out_path.exists():
        save_admiral_checkpoint(out_path, model, None, 0)

    return {
        "out_path": str(out_path),
        "games": len(chunks),
        "samples": total_samples,
        "val_init": val_init,
        "val_best": best_val_metrics,
    }


def main():
    p = argparse.ArgumentParser(
        description="Supervised .sbd multi-task pre-trainer for AdmiralNet"
    )
    p.add_argument(
        "--sbd-dirs",
        nargs="+",
        required=True,
        help="Directories (or files) containing SBD1 .sbd game recordings",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/admiral_belief.pt",
        help="Output pre-trained AdmiralNet checkpoint path",
    )
    p.add_argument("--max-games", type=int, default=600)
    p.add_argument("--samples-per-game", type=int, default=24)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pretrain_admiral_from_sbd(
        sbd_dirs=args.sbd_dirs,
        out_path=args.out,
        max_games=args.max_games,
        samples_per_game=args.samples_per_game,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        workers=args.workers,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
