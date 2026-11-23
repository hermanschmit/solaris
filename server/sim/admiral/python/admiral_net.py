"""Heterogeneous Star+Player Graph Transformer & Distributional Critic (`AdmiralNet`).

Key architectural capabilities:
1. `log1p`-scaled inputs across all ship, infrastructure, credit, and marginal
   upgrade-cost features (stable in 3,000-tick late-game wars).
2. Explicit Pairwise Star-Graph Attention Biases (`[B, H, S, S]`): Euclidean
   travel ticks, multi-hop hyperspace shortest-path ticks, direct jump reach,
   wormholes/warp-gates, and shared ownership.
3. Coupled Star <-> Player Graph Transformer (`backbone` & `critic_backbone`),
   with `forward_pretrain` multi-task heads for offline `.sbd` supervised
   pre-training (4-channel conservation belief, star threat/incoming fleets,
   opponent hidden credits/tokens/research, fog & truth rank critics, and
   2-cycle future star control).
4. Expressive Per-Part Clipped Policy Heads (`PART_NAMES`):
   - `frontier` (`[B, S]` continuous log-weights over reachable frontier stars,
     giving fine-grained per-star claim & Time-on-Target invasion scoring),
   - `posture` (`balanced`, `expand`, `conquer`, `fortify`, `warp_surge`),
   - `target_player` (`P`-way rival empire focus with neighbor proximity prior),
   - `budget` (5-way Dirichlet `[sci, ind, eco, wg, carrierCash]`),
   - `knobs` (3 independent continuous knobs `[attackMargin, maxAttackTicks, triageThreshold]`),
   - `research` (per-technology feature-conditioned head without artificial
     terraforming caps), and `spec` + `spec_star`.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet, Normal

_UPSTREAM_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python")
)
if _UPSTREAM_PY not in sys.path:
    sys.path.insert(0, _UPSTREAM_PY)

import belief_data as bd  # noqa: E402
from belief_tracker import (  # noqa: E402
    HIDDEN,
    N_DELTA_FEATURES,
    VISIBLE,
    BeliefTracker,
)

MAX_PLAYERS = 16
MAX_SPECIALIST_ID = 18
MAX_FRONTIER_STARS = 32
N_EDGE_FEATURES = 5
N_BASE_PLAYER = 16  # before tech levels (T) and delta features (10)
D_TECH_IN = 6

POSTURES = ("balanced", "expand", "conquer", "fortify", "warp_surge")

# Default cycle budget shares: [science, industry, economy, warpGate, carrierCash]
DEFAULT_BUDGET_ALPHA = [2.4, 3.2, 3.5, 0.4, 0.5]

# Continuous tactical knobs: (name, default, scale, lo, hi)
TACTICAL_KNOBS = [
    ("attackMargin", 3.2, 0.35, 1.15, 6.0),
    ("maxAttackTicks", 42.0, 0.30, 14.0, 100.0),
    ("triageThreshold", 0.85, 0.20, 0.35, 1.25),
]

SPECIALIST_PRIOR = {
    11: (7.5, 2),  # Research Station -> 2x science output (cost 5 tokens)
    2: (5.8, 1),   # Space Dock -> +2 manufacturing, -1 terraforming (cost 2 tokens)
    4: (5.0, 0),   # Orbital Cannon -> +1 defender bonus (cost 2 tokens)
}

TECH_BASE_WEIGHTS = {
    "weapons": 2.70,
    "manufacturing": 2.65,
    "hyperspace": 2.20,
    "terraforming": 2.15,
    "banking": 1.85,
    "scanning": 1.65,
    "experimentation": 0.55,
    "specialists": 0.45,
}
CORE_TECH_NAMES = ("weapons", "manufacturing")

KEEP_BIAS = 3.5
INITIAL_KNOB_STD = 0.22
INITIAL_FRONTIER_STD = 0.28

HEAD_GROUPS = ("spatial", "budget", "knobs", "macro")
PART_NAMES = (
    "frontier",
    "posture",
    "target_player",
    "budget",
    "knobs",
    "research",
    "spec",
    "spec_star",
)


def zero_last(module: nn.Linear) -> nn.Linear:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def mlp(n_in: int, n_out: int, hidden: int = 128, zero_out: bool = True) -> nn.Sequential:
    last = nn.Linear(hidden, n_out)
    if zero_out:
        zero_last(last)
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), last)


def placement_rewards_vector(n_players: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Rewards `[+1.0, ..., -1.0]` for ranks `1..n_players`."""
    if n_players <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    ranks = torch.arange(n_players, device=device, dtype=dtype)
    return 1.0 - 2.0 * ranks / float(n_players - 1)


def build_admiral_inputs(
    static: bd.Static,
    frame: dict,
    seat: int,
    own_facts: dict | None = None,
    tracker: BeliefTracker | None = None,
    ticks_to_end: int | None = None,
    truth_facts: dict | None = None,
) -> dict[str, np.ndarray]:
    """Featurize an Admiral observation (or privileged truth state) into
    `log1p`-scaled NumPy arrays for `AdmiralNet`."""
    S, P = static.S, static.P
    T = len(static.techs)
    cycle = max(1, int(static.cycle))
    tick = int(frame["tick"])

    owner = np.asarray(frame["owner"], dtype=np.int64)
    vis = np.asarray(frame["visibility"], dtype=np.int32)
    seen_tick = np.asarray(frame["seen_tick"], dtype=np.int32)
    mem_extras = np.asarray(frame["mem_extras"], dtype=np.float32)  # [S, 2]: wg, spec
    levels = np.asarray(frame["levels"], dtype=np.float32)  # [P, T]
    stats = np.asarray(frame["stats"], dtype=np.float32)  # [P, 6]
    defeated = np.asarray(frame["defeated"], dtype=np.float32)  # [P]

    if tracker is not None:
        conserved = tracker.constrain_beliefs(frame, include_transit_share=False)
        deltas = tracker.delta_features(frame)
    else:
        tmp = BeliefTracker(static, seat)
        conserved = tmp.constrain_beliefs(frame, include_transit_share=False)
        deltas = tmp.delta_features(frame)

    ships_est = conserved["ships"]
    eco_est = conserved["economy"]
    ind_est = conserved["industry"]
    sci_est = conserved["science"]

    own = (owner == seat).astype(np.float32)
    enemy = ((owner >= 0) & (owner != seat)).astype(np.float32)
    unowned = (owner < 0).astype(np.float32)

    age_cycles = np.where(
        seen_tick >= 0,
        np.maximum(0.0, (tick - seen_tick) / float(cycle)),
        50.0,
    ).astype(np.float32)

    # Own & enemy carrier fleet presence per star
    own_orbit = np.zeros(S, dtype=np.float32)
    own_inbound = np.zeros(S, dtype=np.float32)
    enemy_inbound = np.zeros(S, dtype=np.float32)
    enemy_min_eta = np.full(S, float(cycle * 3), dtype=np.float32)

    car = frame["carriers"]
    c_owner = np.asarray(car["owner"], dtype=np.int32)
    c_ships = np.maximum(np.asarray(car["ships"], dtype=np.float32), 0.0)
    c_src = np.asarray(car["source"], dtype=np.int32)
    c_dst = np.asarray(car["destination"], dtype=np.int32)
    c_x = np.asarray(car["x"], dtype=np.float32)
    c_y = np.asarray(car["y"], dtype=np.float32)

    for idx in range(len(c_owner)):
        q = int(c_owner[idx])
        dst = int(c_dst[idx])
        src = int(c_src[idx])
        sh = float(c_ships[idx])
        if q == seat:
            if dst < 0 and 0 <= src < S:
                own_orbit[src] += sh
            elif 0 <= dst < S:
                own_inbound[dst] += sh
        else:
            if 0 <= dst < S:
                enemy_inbound[dst] += max(sh, 1.0)
                eta = np.hypot(
                    float(static.xy[dst, 0] - c_x[idx]),
                    float(static.xy[dst, 1] - c_y[idx]),
                ) / max(static.speed, 1e-4)
                if eta < enemy_min_eta[dst]:
                    enemy_min_eta[dst] = eta

    # Upgrade cost calculator features on own stars
    eco_cost = np.zeros(S, dtype=np.float32)
    ind_cost = np.zeros(S, dtype=np.float32)
    sci_cost = np.zeros(S, dtype=np.float32)
    wg_cost = np.zeros(S, dtype=np.float32)
    eco_aff = np.zeros(S, dtype=np.float32)
    ind_aff = np.zeros(S, dtype=np.float32)
    sci_aff = np.zeros(S, dtype=np.float32)
    wg_aff = np.zeros(S, dtype=np.float32)

    credits = float(own_facts["credits"]) if own_facts else 0.0
    spec_budget = float(own_facts["specialistBudget"]) if own_facts else 0.0
    if own_facts and "upgradeCosts" in own_facts:
        uc = own_facts["upgradeCosts"]
        for arr_raw, out_c, out_a, scale in (
            (uc["economy"], eco_cost, eco_aff, 6.0),
            (uc["industry"], ind_cost, ind_aff, 6.0),
            (uc["science"], sci_cost, sci_aff, 6.0),
            (uc["warpGate"], wg_cost, wg_aff, 7.0),
        ):
            raw = np.asarray(arr_raw, dtype=np.float32)
            valid = raw > 0
            out_c[valid] = np.log1p(raw[valid]) / scale
            out_a[valid] = (raw[valid] <= credits).astype(np.float32)

    # Shortest paths & hyperspace range features
    own_hs_lvl = int(static.hyperspace(levels, seat))
    own_hs_ticks = float(
        static.hyper_range[np.clip(own_hs_lvl, 1, len(static.hyper_range)) - 1]
        / max(static.speed, 1e-4)
    )
    far_hs_ticks = float(
        static.hyper_range[np.clip(own_hs_lvl + 3, 1, len(static.hyper_range)) - 1]
        / max(static.speed, 1e-4)
    )
    own_paths = static.paths(own_hs_lvl)  # [S, S] travel ticks

    own_idx = np.flatnonzero(owner == seat)
    enemy_idx = np.flatnonzero((owner >= 0) & (owner != seat))
    max_t = float(cycle * 6)
    ticks_to_own = (
        np.minimum(own_paths[:, own_idx].min(axis=1), max_t)
        if len(own_idx) > 0
        else np.full(S, max_t, dtype=np.float32)
    )
    euclid_to_own = (
        static.travel[:, own_idx].min(axis=1)
        if len(own_idx) > 0
        else np.full(S, max_t, dtype=np.float32)
    )
    ticks_to_enemy = (
        np.minimum(static.travel[:, enemy_idx].min(axis=1), max_t)
        if len(enemy_idx) > 0
        else np.full(S, max_t, dtype=np.float32)
    )
    in_own_jump = (
        (euclid_to_own <= own_hs_ticks).astype(np.float32)
        if len(own_idx) > 0
        else np.zeros(S, dtype=np.float32)
    )

    # Active frontier candidate stars [S] (reachable claims + reachable enemy border stars)
    frontier_mask = np.zeros(S, dtype=bool)
    if len(own_idx) > 0:
        cand_unowned = (owner < 0) & (
            (ticks_to_own <= float(cycle * 2.5)) | (euclid_to_own <= far_hs_ticks)
        )
        cand_enemy = ((owner >= 0) & (owner != seat)) & (
            (euclid_to_own <= own_hs_ticks * 1.05) | (ticks_to_own <= float(cycle * 1.8))
        )
        cand_all = np.flatnonzero(cand_unowned | cand_enemy)
        if len(cand_all) > MAX_FRONTIER_STARS:
            order_f = np.argsort(ticks_to_own[cand_all])[:MAX_FRONTIER_STARS]
            cand_all = cand_all[order_f]
        frontier_mask[cand_all] = True

    hs_all = np.array(
        [int(static.hyperspace(levels, q)) for q in range(P)], dtype=np.int64
    )
    enemy_r_ticks = (
        static.hyper_range[np.clip(hs_all, 1, len(static.hyper_range)) - 1]
        / max(static.speed, 1e-4)
    )
    in_enemy_jump = np.zeros(S, dtype=np.float32)
    if len(enemy_idx) > 0:
        in_enemy_jump = (
            static.travel[enemy_idx] <= enemy_r_ticks[owner[enemy_idx]][:, None]
        ).any(axis=0).astype(np.float32)

    star_core = np.stack(
        [
            static.xy[:, 0] / 100.0,
            static.xy[:, 1] / 100.0,
            own,
            enemy,
            unowned,
            (static.home >= 0).astype(np.float32),
            (vis == VISIBLE).astype(np.float32),
            (vis != HIDDEN).astype(np.float32),
            (seen_tick >= 0).astype(np.float32),
            np.log1p(age_cycles) / 4.0,
            np.log1p(np.maximum(ships_est, 0.0)) / 6.0,
            np.log1p(np.maximum(eco_est, 0.0)) / 3.5,
            np.log1p(np.maximum(ind_est, 0.0)) / 3.5,
            np.log1p(np.maximum(sci_est, 0.0)) / 3.5,
            mem_extras[:, 0],  # warp gate
            mem_extras[:, 1],  # specialist
            np.log1p(own_orbit) / 5.0,
            np.log1p(own_inbound) / 5.0,
            np.log1p(enemy_inbound) / 5.0,
            np.log1p(enemy_min_eta) / 4.0,
            eco_cost,
            eco_aff,
            ind_cost,
            ind_aff,
            sci_cost,
            sci_aff,
            wg_cost,
            wg_aff,
            np.log1p(ticks_to_own) / 4.0,
            np.log1p(ticks_to_enemy) / 4.0,
            in_own_jump,
            in_enemy_jump,
        ],
        axis=1,
    ).astype(np.float32)  # [S, 32]

    star_x = np.concatenate([star_core, static.quality], axis=1).astype(np.float32)

    # Pairwise Star Edge Features [S, S, N_EDGE_FEATURES=5]
    euclid_ticks = static.travel.astype(np.float32)
    hop_ticks = np.minimum(own_paths, max_t).astype(np.float32)
    direct_jump = (euclid_ticks <= own_hs_ticks).astype(np.float32)
    wg_vec = mem_extras[:, 0] > 0.5
    fast_link = np.outer(wg_vec, wg_vec).astype(np.float32)
    same_owner = ((owner[:, None] == owner[None, :]) & (owner[:, None] >= 0)).astype(
        np.float32
    )
    edge_x = np.stack(
        [
            np.log1p(euclid_ticks) / 4.5,
            np.log1p(hop_ticks) / 5.0,
            direct_jump,
            fast_link,
            same_owner,
        ],
        axis=-1,
    ).astype(np.float32)

    # Player Token Features [P, N_BASE_PLAYER + T + N_DELTA_FEATURES], ordered [seat, others...]
    p_order = [seat] + [q for q in range(P) if q != seat]
    perm_inv = np.zeros(P, dtype=np.int64)
    for rank_idx, q in enumerate(p_order):
        perm_inv[q] = rank_idx

    owner_rel = np.where(owner >= 0, perm_inv[np.maximum(owner, 0)], -1).astype(
        np.int64
    )

    tot_stars = max(1.0, float(stats[:, 0].sum()))
    tot_ships = max(1.0, float(stats[:, 1].sum()))

    alive_enemy_mask = np.zeros(P, dtype=bool)
    alive_enemy_mask[0] = True  # index 0 = opportunistic / no fixed rival

    player_rows = []
    for idx_in_order, q in enumerate(p_order):
        is_self = 1.0 if q == seat else 0.0
        st = stats[q]
        lvl = levels[q] / 10.0

        if idx_in_order >= 1 and defeated[q] <= 0.5 and st[0] > 0:
            alive_enemy_mask[idx_in_order] = True

        q_stars = owner == q
        if q == seat:
            min_dist_own = 0.0
        elif q_stars.any() and len(own_idx) > 0:
            min_dist_own = float(np.minimum(euclid_to_own[q_stars].min(), max_t))
        else:
            min_dist_own = max_t

        if truth_facts is not None:
            p_cred = float(truth_facts["playerCredits"][q])
            p_spec = float(truth_facts["playerSpecTokens"][q])
            r_now = int(truth_facts["playerResearchingNow"][q])
            r_prog_vec = truth_facts["playerResearchProgress"][q]
            r_prog = float(r_prog_vec[r_now]) if 0 <= r_now < T else 0.0
            r_req = (
                float(own_facts["researchRequired"][r_now])
                if (own_facts and q == seat and 0 <= r_now < T)
                else max(100.0, float(levels[q, max(r_now, 0)] * 100.0))
            )
        elif q == seat and own_facts is not None:
            p_cred = credits
            p_spec = spec_budget
            r_now = int(own_facts["researchingNow"])
            r_prog = (
                float(own_facts["researchProgress"][r_now])
                if 0 <= r_now < T
                else 0.0
            )
            r_req = (
                float(own_facts["researchRequired"][r_now])
                if 0 <= r_now < T
                else 1.0
            )
        else:
            p_cred = 0.0
            p_spec = 0.0
            r_now = -1
            r_prog = 0.0
            r_req = 1.0

        r_frac = np.clip(r_prog / max(r_req, 1.0), 0.0, 1.0)
        sci_pts = max(1.0, float(st[5]))
        eta_res = max(0.0, (r_req - r_prog) / sci_pts) if r_req > 0 else 0.0

        base_p = np.array(
            [
                is_self,
                float(defeated[q]),
                np.log1p(max(st[0], 0.0)) / 5.0,
                np.log1p(max(st[1], 0.0)) / 7.0,
                np.log1p(max(st[2], 0.0)) / 4.0,
                np.log1p(max(st[3], 0.0)) / 5.0,
                np.log1p(max(st[4], 0.0)) / 5.0,
                np.log1p(max(st[5], 0.0)) / 4.5,
                float(st[0]) / tot_stars,
                float(st[1]) / tot_ships,
                np.log1p(max(p_cred, 0.0)) / 8.0,
                np.log1p(max(p_spec, 0.0)) / 4.0,
                r_frac,
                np.log1p(eta_res) / 5.0,
                float(r_now + 1) / float(T + 1),
                np.log1p(min_dist_own) / 4.5,
            ],
            dtype=np.float32,
        )  # length 16 == N_BASE_PLAYER
        player_rows.append(np.concatenate([base_p, lvl.astype(np.float32), deltas[q]]))

    player_x = np.stack(player_rows, axis=0).astype(np.float32)

    # Action masks
    own_mask = owner == seat  # [S]
    non_own_mask = owner != seat  # [S]

    research_mask = np.zeros(1 + T, dtype=bool)
    research_mask[0] = True
    for ti, t_name in enumerate(static.techs):
        if t_name in static.researchable:
            research_mask[1 + ti] = True

    # Per-technology calculator & light iter-0 tie-breaker features [T, D_TECH_IN=6]
    # Note: `weapons` and `manufacturing` are NOT penalized for advancing ahead
    # of `terraforming`, allowing late-game weapon/manufacturing scaling.
    own_lvls = levels[seat].astype(np.float32)
    alive_rivals = [
        q for q in range(P) if q != seat and defeated[q] <= 0.5 and stats[q, 0] > 0
    ]
    max_enemy_lvls = (
        levels[alive_rivals].max(axis=0).astype(np.float32)
        if alive_rivals
        else own_lvls.copy()
    )
    res_able = research_mask[1:]
    core_able = np.array(
        [
            (t_name in CORE_TECH_NAMES) and bool(res_able[ti])
            for ti, t_name in enumerate(static.techs)
        ],
        dtype=bool,
    )
    if core_able.any():
        min_wm_lvl = float(own_lvls[core_able].min())
    elif res_able.any():
        min_wm_lvl = float(own_lvls[res_able].min())
    else:
        min_wm_lvl = 1.0

    tech_x = np.zeros((T, D_TECH_IN), dtype=np.float32)
    for ti, t_name in enumerate(static.techs):
        prog_t = (
            float(own_facts["researchProgress"][ti])
            if (own_facts and ti < len(own_facts["researchProgress"]))
            else 0.0
        )
        req_t = (
            float(own_facts["researchRequired"][ti])
            if (own_facts and ti < len(own_facts["researchRequired"]))
            else max(100.0, float(own_lvls[ti]) * 100.0)
        )
        frac_t = float(np.clip(prog_t / max(req_t, 1.0), 0.0, 1.0)) if req_t > 0 else 0.0
        lvl_t = float(own_lvls[ti])

        base_w = TECH_BASE_WEIGHTS.get(t_name, 1.0)
        if t_name == "hyperspace":
            if lvl_t < 4.0:
                base_w += 0.95 * (4.0 - lvl_t) / 3.0
            else:
                base_w -= 0.70 * (lvl_t - 3.0)
        elif t_name == "terraforming":
            if lvl_t >= 3.0:
                base_w -= 0.65 * (lvl_t - 2.0)
        elif t_name == "banking":
            if lvl_t >= 3.0:
                base_w -= 0.70 * (lvl_t - 2.0)
        elif t_name == "scanning":
            if lvl_t < 3.0:
                base_w += 0.70 * (3.0 - lvl_t) / 2.0
            else:
                base_w -= 0.55 * (lvl_t - 2.0)
        elif t_name == "weapons":
            base_w += 0.45 * max(0.0, float(max_enemy_lvls[ti] - lvl_t))

        ahead_wm = max(0.0, lvl_t - min_wm_lvl) if t_name in CORE_TECH_NAMES else max(0.0, lvl_t - 2.0)
        prior_t = base_w - 0.35 * ahead_wm + 2.2 * frac_t

        tech_x[ti] = np.array(
            [
                lvl_t / 10.0,
                (lvl_t - min_wm_lvl) / 4.0,
                float(max_enemy_lvls[ti] - lvl_t) / 4.0,
                np.log1p(max(req_t, 0.0)) / 9.0,
                frac_t,
                prior_t,
            ],
            dtype=np.float32,
        )

    spec_mask = np.zeros(1 + MAX_SPECIALIST_ID, dtype=bool)
    spec_mask[0] = True
    free_own_star = own_mask & (mem_extras[:, 1] < 0.5)
    if own_facts and free_own_star.any():
        for sid in own_facts.get("affordableStarSpecialists", []):
            if 1 <= int(sid) <= MAX_SPECIALIST_ID:
                spec_mask[int(sid)] = True

    # Global timing vector [4]
    cycle_phase = float(tick % cycle) / float(cycle)
    game_progress = np.log1p(float(tick)) / 8.0
    end_feat = (
        float(ticks_to_end) / 400.0 if ticks_to_end is not None else 0.0
    )
    has_end = 1.0 if ticks_to_end is not None else 0.0
    time_x = np.array(
        [cycle_phase, game_progress, end_feat, has_end], dtype=np.float32
    )

    return {
        "star_x": star_x,                       # [S, 41]
        "edge_x": edge_x,                       # [S, S, 5]
        "player_x": player_x,                   # [P, 16 + T + 10]
        "tech_x": tech_x,                       # [T, 6]
        "owner_rel": owner_rel,                 # [S] in {-1, 0..P-1}
        "time_x": time_x,                       # [4]
        "own_mask": own_mask,                   # [S]
        "non_own_mask": non_own_mask,           # [S]
        "frontier_mask": frontier_mask,         # [S]
        "free_own_mask": free_own_star,         # [S]
        "alive_enemy_mask": alive_enemy_mask,   # [P]
        "research_mask": research_mask,         # [1 + T]
        "spec_mask": spec_mask,                 # [1 + MAX_SPECIALIST_ID]
    }


D_STAR_TOTAL = 32 + bd.N_QUALITY  # 41


class BiasedStarAttention(nn.Module):
    """Multi-head star self-attention with pairwise relational bias `[B, H, S, S]`."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, S, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=edge_bias)
        return self.proj(out.transpose(1, 2).contiguous().view(B, S, D))


class HeterogeneousLayer(nn.Module):
    """One layer of coupled Star <-> Player Graph Transformer."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.norm_s1 = nn.LayerNorm(d)
        self.star_attn = BiasedStarAttention(d, n_heads)
        self.norm_s2 = nn.LayerNorm(d)
        self.star_ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

        self.star_to_player = nn.Linear(2 * d, d)
        self.norm_p1 = nn.LayerNorm(d)
        self.player_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_p2 = nn.LayerNorm(d)
        self.player_ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )
        self.player_to_star = nn.Linear(d, d)

    def forward(
        self,
        h_s: torch.Tensor,
        h_p: torch.Tensor,
        edge_bias: torch.Tensor,
        owner_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = h_s.shape
        P = h_p.shape[1]

        # 1. Star -> Star relational self-attention
        h_s = h_s + self.star_attn(self.norm_s1(h_s), edge_bias)

        # 2. Pool owned stars -> Player tokens
        onehot = F.one_hot(owner_rel.clamp(min=-1) + 1, P + 1)[..., 1:].to(
            h_s.dtype
        )  # [B, S, P]
        counts = onehot.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B, P, 1]
        s_mean = torch.einsum("bsp,bsd->bpd", onehot, h_s) / counts
        neg_inf = torch.full_like(h_s[:, :, :1], -1e4)
        expanded_mask = onehot.unsqueeze(-1) > 0.5  # [B, S, P, 1]
        s_max = torch.where(
            expanded_mask,
            h_s.unsqueeze(2),
            neg_inf.unsqueeze(2),
        ).max(dim=1).values
        s_max = torch.where(
            (onehot.sum(dim=1, keepdim=False) > 0).unsqueeze(-1),
            s_max,
            torch.zeros_like(s_max),
        )
        h_p = h_p + self.star_to_player(torch.cat([s_mean, s_max], dim=-1))

        # 3. Player -> Player geopolitical self-attention
        p_norm = self.norm_p1(h_p)
        p_attn_out, _ = self.player_attn(p_norm, p_norm, p_norm, need_weights=False)
        h_p = h_p + p_attn_out
        h_p = h_p + self.player_ff(self.norm_p2(h_p))

        # 4. Broadcast Player context back to owned Star tokens + Star FFN
        padded_p = torch.cat(
            [torch.zeros(B, 1, D, device=h_p.device, dtype=h_p.dtype), h_p],
            dim=1,
        )  # index 0 = unowned
        gather_idx = (owner_rel.clamp(min=-1) + 1).unsqueeze(-1).expand(B, S, D)
        owner_ctx = padded_p.gather(1, gather_idx)
        h_s = h_s + self.player_to_star(owner_ctx)
        h_s = h_s + self.star_ff(self.norm_s2(h_s))
        return h_s, h_p


class HeterogeneousBackbone(nn.Module):
    def __init__(self, n_techs: int, d: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        d_player_in = N_BASE_PLAYER + n_techs + N_DELTA_FEATURES
        self.star_proj = nn.Linear(D_STAR_TOTAL, d)
        self.player_proj = nn.Linear(d_player_in, d)
        self.time_proj = nn.Linear(4, d)
        self.edge_proj = nn.Sequential(
            nn.Linear(N_EDGE_FEATURES, 16),
            nn.GELU(),
            nn.Linear(16, n_heads),
        )
        self.layers = nn.ModuleList(
            [HeterogeneousLayer(d, n_heads) for _ in range(n_layers)]
        )
        self.out_norm_s = nn.LayerNorm(d)
        self.out_norm_p = nn.LayerNorm(d)

    def forward(
        self,
        star_x: torch.Tensor,
        edge_x: torch.Tensor,
        player_x: torch.Tensor,
        owner_rel: torch.Tensor,
        time_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_emb = self.time_proj(time_x)  # [B, D]
        h_s = self.star_proj(star_x) + t_emb.unsqueeze(1)
        h_p = self.player_proj(player_x) + t_emb.unsqueeze(1)
        edge_bias = self.edge_proj(edge_x).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            h_s, h_p = layer(h_s, h_p, edge_bias, owner_rel)

        h_s = self.out_norm_s(h_s)
        h_p = self.out_norm_p(h_p)

        p_self = h_p[:, 0]
        p_mean = h_p.mean(dim=1)
        p_max = h_p.max(dim=1).values
        g = torch.cat([p_self, p_mean, p_max, t_emb], dim=-1)  # [B, 4 * D]
        return h_s, h_p, g


class DistributionalValueHead(nn.Module):
    """Predicts placement logits `[B, MAX_PLAYERS]` + cycle shaping bonus `[B]`."""

    def __init__(self, g_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, 128),
            nn.GELU(),
            nn.Linear(128, MAX_PLAYERS),
        )
        self.bonus_head = mlp(g_dim, 1, hidden=64, zero_out=True)

    def forward(self, g: torch.Tensor, n_players: int):
        logits = self.net(g)[:, :n_players]  # [B, P]
        probs = F.softmax(logits, dim=-1)
        rewards = placement_rewards_vector(n_players, device=g.device, dtype=g.dtype)
        expected_place_val = (probs * rewards.unsqueeze(0)).sum(dim=-1)
        bonus_val = 0.5 * torch.tanh(self.bonus_head(g).squeeze(-1))
        return logits, expected_place_val + bonus_val, expected_place_val


class AdmiralNet(nn.Module):
    def __init__(self, n_techs: int, d: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.n_techs = n_techs
        self.d = d
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.backbone = HeterogeneousBackbone(n_techs, d=d, n_layers=n_layers, n_heads=n_heads)
        self.critic_backbone = HeterogeneousBackbone(
            n_techs, d=d, n_layers=n_layers, n_heads=n_heads
        )
        g_dim = 4 * d

        # --- Group 1: Fine-Grained Per-Frontier-Star Scoring + Stance + Rival ---
        self.frontier_head = mlp(d + g_dim, 1, hidden=128, zero_out=True)
        self.frontier_log_std = nn.Parameter(
            torch.tensor(math.log(INITIAL_FRONTIER_STD))
        )
        self.posture_head = mlp(g_dim, len(POSTURES))
        self.target_none_head = mlp(g_dim, 1)
        self.target_emp_head = mlp(d + g_dim, 1)

        with torch.no_grad():
            self.posture_head[-1].bias[0] = 1.8
            self.target_none_head[-1].bias.fill_(1.6)

        # --- Group 2: Cycle Budget Dirichlet Head [5] ---
        self.budget_head = mlp(g_dim, 5)
        self.register_buffer(
            "default_alpha", torch.tensor(DEFAULT_BUDGET_ALPHA, dtype=torch.float32)
        )

        # --- Group 3: Tactical Continuous Knobs [3] ---
        self.knob_head = mlp(g_dim, len(TACTICAL_KNOBS))
        self.knob_log_std = nn.Parameter(
            torch.full((len(TACTICAL_KNOBS),), math.log(INITIAL_KNOB_STD))
        )

        # --- Group 4: Discrete Macro Heads (Feature-Conditioned Research + Specialist) ---
        self.research_keep_head = mlp(g_dim, 1)
        self.research_tech_head = mlp(g_dim + D_TECH_IN, 1, hidden=96)
        self.spec_head = mlp(g_dim, 1 + MAX_SPECIALIST_ID)
        self.spec_star_head = mlp(d, MAX_SPECIALIST_ID)
        with torch.no_grad():
            self.research_keep_head[-1].bias.fill_(0.5)
            self.spec_head[-1].bias[0] = KEEP_BIAS

        # Auxiliary conservation-constrained hidden belief head:
        # [ships_logit, eco_logit, ind_logit, sci_logit] per star
        self.belief_head = mlp(d, 4)

        # Auxiliary dense representation heads trained during `.sbd` pre-training:
        # - star tactical threat `[log1p(true_ships), log1p(threat_1c), log1p(incoming)]`
        # - opponent hidden state `[log1p(credits)/8, log1p(spec_tokens)/4, researchingNow_logits...]`
        # - privileged truth 2-cycle star ownership projection
        self.aux_star_threat_head = zero_last(nn.Linear(d, 3))
        self.aux_opp_state_head = zero_last(nn.Linear(d, 2 + n_techs))
        self.aux_truth_expand_head = zero_last(nn.Linear(d, 1))

        # Dual Distributional Critics:
        self.fog_critic = DistributionalValueHead(g_dim)
        self.truth_critic = DistributionalValueHead(g_dim)

    def trunk_parameters(self) -> list[nn.Parameter]:
        """Parameters of the pre-trained fog Graph Transformer + belief head
        (updated with a smaller learning rate during PPO so representations stay sharp)."""
        return list(self.backbone.parameters()) + list(self.belief_head.parameters())

    def forward_pretrain(
        self,
        fog_batch: dict[str, torch.Tensor],
        truth_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Multi-task forward pass for offline `.sbd` supervised pre-training of
        both `backbone` (fog) and `critic_backbone` (privileged truth)."""
        P = fog_batch["player_x"].shape[1]
        h_s_fog, h_p_fog, g_fog = self.backbone(
            fog_batch["star_x"],
            fog_batch["edge_x"],
            fog_batch["player_x"],
            fog_batch["owner_rel"],
            fog_batch["time_x"],
        )
        belief_logits = self.belief_head(h_s_fog)
        raw_threat = self.aux_star_threat_head(h_s_fog)
        base_ships = fog_batch["star_x"][:, :, 10:11] * 1.2  # log1p(ships_est) / 5.0
        base_in = fog_batch["star_x"][:, :, 18:19]           # log1p(seen_enemy_in) / 5.0
        star_threat_pred = torch.cat(
            [
                raw_threat[:, :, 0:1] + base_ships,
                raw_threat[:, :, 1:2] + base_ships,
                raw_threat[:, :, 2:3] + base_in,
            ],
            dim=-1,
        )
        opp_state_pred = self.aux_opp_state_head(h_p_fog)
        fog_rank_logits, fog_val, _ = self.fog_critic(g_fog, P)

        h_s_truth, _, g_truth = self.critic_backbone(
            truth_batch["star_x"],
            truth_batch["edge_x"],
            truth_batch["player_x"],
            truth_batch["owner_rel"],
            truth_batch["time_x"],
        )
        truth_rank_logits, truth_val, _ = self.truth_critic(g_truth, P)
        truth_expand_logits = self.aux_truth_expand_head(h_s_truth).squeeze(-1)

        return {
            "belief_logits": belief_logits,
            "star_threat_pred": star_threat_pred,
            "opp_state_pred": opp_state_pred,
            "fog_rank_logits": fog_rank_logits,
            "fog_val": fog_val,
            "truth_rank_logits": truth_rank_logits,
            "truth_val": truth_val,
            "truth_expand_logits": truth_expand_logits,
        }

    def forward_policy(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        star_x = batch["star_x"]
        edge_x = batch["edge_x"]
        player_x = batch["player_x"]
        tech_x = batch["tech_x"]
        owner_rel = batch["owner_rel"]
        time_x = batch["time_x"]
        own_mask = batch["own_mask"]
        frontier_mask = batch["frontier_mask"]
        free_own_mask = batch["free_own_mask"]
        alive_enemy_mask = batch["alive_enemy_mask"]
        research_mask = batch["research_mask"]
        spec_mask = batch["spec_mask"]

        B, S, _ = star_x.shape
        P = player_x.shape[1]
        T = tech_x.shape[1]

        h_s, h_p, g = self.backbone(star_x, edge_x, player_x, owner_rel, time_x)

        # 1. Per-Frontier-Star Continuous Scoring [B, S]
        f_inp = torch.cat([h_s, g.unsqueeze(1).expand(B, S, -1)], dim=-1)
        frontier_mu = torch.tanh(self.frontier_head(f_inp).squeeze(-1)) * 2.0
        frontier_mu = torch.where(frontier_mask, frontier_mu, torch.zeros_like(frontier_mu))
        frontier_std = self.frontier_log_std.clamp(-2.8, 0.2).exp().expand_as(frontier_mu)

        # 2. Strategic Posture [B, 5] & Target Empire [B, P]
        posture_logits = self.posture_head(g)
        t_none = self.target_none_head(g)
        if P > 1:
            emp_inp = torch.cat(
                [h_p[:, 1:], g.unsqueeze(1).expand(B, P - 1, -1)], dim=-1
            )
            neighbor_prior = -0.85 * player_x[:, 1:, 15]
            t_emps = self.target_emp_head(emp_inp).squeeze(-1) + neighbor_prior
            target_logits = torch.cat([t_none, t_emps], dim=-1)
        else:
            target_logits = t_none
        target_logits = target_logits.masked_fill(~alive_enemy_mask, -1e9)

        # 3. Dirichlet Cycle Budget [B, 5]: [sci, ind, eco, wg, carrierCash]
        budget_raw = self.budget_head(g)
        budget_alpha = (
            F.softplus(budget_raw + torch.log(torch.expm1(self.default_alpha * 3.0)))
            + 0.2
        )

        # 4. Continuous Tactical Knobs [B, 3]
        knob_mean = torch.tanh(self.knob_head(g)) * 2.0
        knob_std = self.knob_log_std.clamp(-3.0, 0.5).exp().expand_as(knob_mean)

        # 5. Feature-Conditioned Research Head [B, 1 + T]
        keep_logit = self.research_keep_head(g)
        tech_inp = torch.cat([g.unsqueeze(1).expand(B, T, -1), tech_x], dim=-1)
        tech_logits = self.research_tech_head(tech_inp).squeeze(-1) + tech_x[:, :, 5]
        res_logits = torch.cat([keep_logit, tech_logits], dim=-1).masked_fill(
            ~research_mask, -1e9
        )

        # 6. Specialist + Star Pointer
        sp_logits = self.spec_head(g)
        sp_prior = torch.zeros_like(sp_logits)
        for sid, (boost, _) in SPECIALIST_PRIOR.items():
            if sid <= MAX_SPECIALIST_ID:
                sp_prior[:, sid] = boost
        sp_logits = (sp_logits + sp_prior).masked_fill(~spec_mask, -1e9)

        sp_star_logits = self.spec_star_head(h_s)
        sp_star_prior = torch.zeros_like(sp_star_logits)
        if 11 <= MAX_SPECIALIST_ID:
            sp_star_prior[:, :, 10] = (
                3.0 * star_x[:, :, 13]
                + 2.5 * star_x[:, :, 34]
                + 1.5 * star_x[:, :, 39]
                - 1.5 * star_x[:, :, 31]
            )
        if 2 <= MAX_SPECIALIST_ID:
            sp_star_prior[:, :, 1] = (
                3.0 * star_x[:, :, 12]
                + 2.5 * star_x[:, :, 33]
                + 1.2 * star_x[:, :, 39]
                + 0.5 * star_x[:, :, 31]
            )
        if 4 <= MAX_SPECIALIST_ID:
            sp_star_prior[:, :, 3] = (
                1.8 * (star_x[:, :, 11] + star_x[:, :, 12] + star_x[:, :, 13])
                + 2.0 * star_x[:, :, 31]
            )
        has_any_free = free_own_mask.any(dim=1, keepdim=True)
        eff_free = torch.where(has_any_free, free_own_mask, own_mask)
        eff_free = torch.where(
            eff_free.any(dim=1, keepdim=True),
            eff_free,
            torch.ones_like(eff_free),
        )
        sp_star_logits = (sp_star_logits + sp_star_prior).masked_fill(
            ~eff_free.unsqueeze(-1), -1e9
        )

        belief_logits = self.belief_head(h_s)  # [B, S, 4]
        fog_rank_logits, fog_value, fog_place_val = self.fog_critic(g, P)

        return {
            "frontier_mu": frontier_mu,
            "frontier_std": frontier_std,
            "posture_logits": posture_logits,
            "target_logits": target_logits,
            "budget_alpha": budget_alpha,
            "knob_mean": knob_mean,
            "knob_std": knob_std,
            "res_logits": res_logits,
            "sp_logits": sp_logits,
            "sp_star_logits": sp_star_logits,
            "belief_logits": belief_logits,
            "fog_rank_logits": fog_rank_logits,
            "fog_value": fog_value,
            "fog_place_val": fog_place_val,
        }

    def forward_truth_critic(self, truth_batch: dict[str, torch.Tensor]):
        P = truth_batch["player_x"].shape[1]
        _, _, g_truth = self.critic_backbone(
            truth_batch["star_x"],
            truth_batch["edge_x"],
            truth_batch["player_x"],
            truth_batch["owner_rel"],
            truth_batch["time_x"],
        )
        return self.truth_critic(g_truth, P)

    def act_or_evaluate(
        self,
        batch: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
        sample: bool = True,
        temperature: float = 1.0,
    ) -> dict[str, any]:
        """Sample actions, pick greedy mode, or evaluate exact per-part `logp_parts`
        (`PART_NAMES`) + `part_masks` for independent per-column PPO clipping."""
        out = self.forward_policy(batch)
        temp = max(float(temperature), 1e-3)
        B = batch["star_x"].shape[0]
        dtype = batch["star_x"].dtype
        dev = batch["star_x"].device
        f_mask = batch["frontier_mask"]  # [B, S] bool
        f_mask_f = f_mask.to(dtype)

        chosen: dict[str, torch.Tensor] = {} if actions is None else dict(actions)

        # 1. Per-Frontier-Star Normal [B, S]
        f_dist = Normal(out["frontier_mu"], out["frontier_std"] * (temp if (actions is None and sample) else 1.0))
        eval_f_dist = Normal(out["frontier_mu"], out["frontier_std"])
        if actions is None:
            f_raw = f_dist.rsample() if sample else out["frontier_mu"]
            f_act = (f_raw * f_mask_f).clamp(-3.0, 3.0)
            chosen["frontier"] = f_act
        else:
            f_act = actions["frontier"]
        lp_frontier_cols = eval_f_dist.log_prob(f_act) * f_mask_f  # [B, S]
        ent_frontier = (eval_f_dist.entropy() * f_mask_f).sum(dim=-1)

        # 2. Posture & Target Empire
        p_logits = out["posture_logits"]
        t_logits = out["target_logits"]
        if actions is None:
            p_act = (
                Categorical(logits=p_logits / temp).sample()
                if sample
                else p_logits.argmax(dim=-1)
            )
            t_act = (
                Categorical(logits=t_logits / temp).sample()
                if sample
                else t_logits.argmax(dim=-1)
            )
            chosen["posture"] = p_act
            chosen["target_player"] = t_act
        else:
            p_act = actions["posture"]
            t_act = actions["target_player"]

        eval_p_dist = Categorical(logits=p_logits)
        eval_t_dist = Categorical(logits=t_logits)
        lp_posture = eval_p_dist.log_prob(p_act)
        lp_target = eval_t_dist.log_prob(t_act)
        ent_posture = eval_p_dist.entropy()
        ent_target = eval_t_dist.entropy()

        logp_spatial = lp_frontier_cols.sum(dim=-1) + lp_posture + lp_target
        ent_spatial = ent_frontier + ent_posture + ent_target

        # 3. Budget Dirichlet [B, 5]
        alpha = out["budget_alpha"]
        dir_dist = Dirichlet(alpha)
        if actions is None:
            b = dir_dist.rsample() if sample else alpha / alpha.sum(dim=-1, keepdim=True)
            b = b.clamp(min=1e-4)
            b = b / b.sum(dim=-1, keepdim=True)
            chosen["budget"] = b
        else:
            b = actions["budget"].clamp(min=1e-4)
            b = b / b.sum(dim=-1, keepdim=True)
        lp_budget = dir_dist.log_prob(b)
        ent_budget = dir_dist.entropy()

        # 4. Tactical Continuous Knobs [B, 3]
        knob_dist = Normal(out["knob_mean"], out["knob_std"])
        if actions is None:
            k_act = knob_dist.rsample() if sample else out["knob_mean"]
            chosen["knobs"] = k_act
        else:
            k_act = actions["knobs"]
        lp_knobs_cols = knob_dist.log_prob(k_act)  # [B, 3]
        logp_knobs = lp_knobs_cols.sum(dim=-1)
        ent_knobs = knob_dist.entropy().sum(dim=-1)

        # 5. Macro Discrete: Research + Specialist + Star
        res_dist = Categorical(logits=out["res_logits"])
        sp_dist = Categorical(logits=out["sp_logits"])
        if actions is None:
            r_act = res_dist.sample() if sample else out["res_logits"].argmax(dim=-1)
            s_act = sp_dist.sample() if sample else out["sp_logits"].argmax(dim=-1)
            chosen["research"] = r_act
            chosen["spec"] = s_act
        else:
            r_act = actions["research"]
            s_act = actions["spec"]

        s_idx = (s_act.clamp(min=1) - 1).view(-1, 1, 1).expand(
            -1, out["sp_star_logits"].shape[1], 1
        )
        sel_star_logits = out["sp_star_logits"].gather(2, s_idx).squeeze(-1)
        sp_star_dist = Categorical(logits=sel_star_logits)
        if actions is None:
            st_act = (
                sp_star_dist.sample()
                if sample
                else sel_star_logits.argmax(dim=-1)
            )
            chosen["spec_star"] = st_act
        else:
            st_act = actions["spec_star"]

        hired_mask = s_act > 0
        hired_f = hired_mask.to(dtype)
        lp_res = res_dist.log_prob(r_act)
        lp_spec = sp_dist.log_prob(s_act)
        lp_spec_star = sp_star_dist.log_prob(st_act) * hired_f

        logp_macro = lp_res + lp_spec + lp_spec_star
        ent_macro = (
            res_dist.entropy()
            + sp_dist.entropy()
            + hired_f * sp_star_dist.entropy()
        )

        ones_1 = torch.ones((B, 1), dtype=torch.bool, device=dev)
        ones_3 = torch.ones((B, len(TACTICAL_KNOBS)), dtype=torch.bool, device=dev)
        logp_parts = {
            "frontier": lp_frontier_cols,             # [B, S]
            "posture": lp_posture.unsqueeze(-1),      # [B, 1]
            "target_player": lp_target.unsqueeze(-1), # [B, 1]
            "budget": lp_budget.unsqueeze(-1),        # [B, 1]
            "knobs": lp_knobs_cols,                   # [B, 3]
            "research": lp_res.unsqueeze(-1),         # [B, 1]
            "spec": lp_spec.unsqueeze(-1),            # [B, 1]
            "spec_star": lp_spec_star.unsqueeze(-1),  # [B, 1]
        }
        part_masks = {
            "frontier": f_mask,
            "posture": ones_1,
            "target_player": ones_1,
            "budget": ones_1,
            "knobs": ones_3,
            "research": ones_1,
            "spec": ones_1,
            "spec_star": hired_mask.unsqueeze(-1),
        }

        logp_groups = {
            "spatial": logp_spatial,
            "budget": lp_budget,
            "knobs": logp_knobs,
            "macro": logp_macro,
        }
        ent_groups = {
            "spatial": ent_spatial,
            "budget": ent_budget,
            "knobs": ent_knobs,
            "macro": ent_macro,
        }
        total_logp = logp_spatial + lp_budget + logp_knobs + logp_macro
        total_ent = ent_spatial + ent_budget + ent_knobs + ent_macro

        return {
            "actions": chosen,
            "logp": total_logp,
            "logp_groups": logp_groups,
            "logp_parts": logp_parts,
            "part_masks": part_masks,
            "entropy_groups": ent_groups,
            "entropy": total_ent,
            "policy_out": out,
        }
