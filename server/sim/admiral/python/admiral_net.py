"""Heterogeneous Star+Player Graph Transformer & Distributional Critic (`AdmiralNet`).

Key architectural upgrades over `Commander`:
1. All ship, infrastructure, credit, and upgrade-cost inputs use `log1p` scaling
   (eliminating late-game activation & critic explosions in 3000-tick wars).
2. Explicit Pairwise Star-Graph Attention Biases (`[B, H, S, S]`): encodes
   Euclidean travel ticks, multi-hop hyperspace shortest-path ticks, direct jump
   reachability, wormholes/warp-gates, and shared ownership.
3. Coupled Star <-> Player Graph Transformer: Star tokens and Empire (Player)
   tokens exchange information every layer, giving both policy and critic a
   symmetric representation of all `P` empires' tech, economy, and war attrition.
4. Calculator Features: exact marginal upgrade costs per star, research progress
   fraction (`progress / required`), ticks to research completion, and scoreboard
   `Delta stats` war-attrition signals from `BeliefTracker`.
5. Factored Low-Variance Policy Heads:
   - `spatial`: 2 offensive target pointers (`attack_1`, `attack_2`), 1 defensive
     bastion pointer (`defend`), and 3 targeted infrastructure pointers
     (`ind_star`, `sci_star`, `wg_star`), all zero-initialized at index 0 ("broad
     algorithmic ROI") with direct gradient flow into every star's embedding.
   - `budget`: 5-way Dirichlet distribution over `[sci, ind, eco, wg, reserve]`.
   - `knobs`: 3-D Gaussian over `[attackMargin, maxAttackTicks, triageThreshold]`.
   - `macro`: `research` (`keep` + `T` techs), `specialist` (`none` + star
     specialist + star pointer), and `diplomacy` (`hold`, `court`, `betray`).
6. Auxiliary Conservation-Belief Head (`[B, S, 4]`): predicts residual star
   logits for hidden `ships`, `economy`, `industry`, and `science`, projected
   through `BeliefTracker`'s exact scoreboard-conservation layer.
7. 10-Rank Distributional Placement Critic (`truth` & `fog`): predicts placement
   probabilities `p(rank = 1..P)` bounded strictly in `[-1, 1]`, plus a bounded
   shaping-bonus head.
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
N_EDGE_FEATURES = 5
D_STAR_IN = 35
N_BASE_PLAYER = 15  # before tech levels (T) and delta features (10)

# Default budget shares: [science, industry, economy, warpGate, carrierReserve]
DEFAULT_BUDGET_ALPHA = [2.2, 3.3, 3.0, 0.6, 1.1]

# Continuous tactical knobs: (name, default, scale, lo, hi)
TACTICAL_KNOBS = [
    ("attackMargin", 1.35, 0.25, 1.02, 3.5),
    ("maxAttackTicks", 42.0, 0.30, 12.0, 120.0),
    ("triageThreshold", 0.85, 0.20, 0.35, 1.25),
]

SPECIALIST_PRIOR = {
    11: (5.5, 2),  # Research Station -> science
    2: (5.0, 1),   # Space Dock -> industry
    4: (4.5, 0),   # Orbital Cannon -> total/economy
}

KEEP_BIAS = 3.0
INITIAL_KNOB_STD = 0.25

HEAD_GROUPS = ("spatial", "budget", "knobs", "macro")


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
    own_paths = static.paths(own_hs_lvl)  # [S, S] travel ticks

    own_idx = np.flatnonzero(owner == seat)
    enemy_idx = np.flatnonzero((owner >= 0) & (owner != seat))
    max_t = float(cycle * 6)
    ticks_to_own = (
        np.minimum(own_paths[:, own_idx].min(axis=1), max_t)
        if len(own_idx) > 0
        else np.full(S, max_t, dtype=np.float32)
    )
    ticks_to_enemy = (
        np.minimum(static.travel[:, enemy_idx].min(axis=1), max_t)
        if len(enemy_idx) > 0
        else np.full(S, max_t, dtype=np.float32)
    )
    in_own_jump = (
        (static.travel[:, own_idx].min(axis=1) <= own_hs_ticks).astype(np.float32)
        if len(own_idx) > 0
        else np.zeros(S, dtype=np.float32)
    )

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

    # Append natural resource & star-type quality [S, 9] -> take first 3 + sum?
    # Better: include full static.quality [S, 9] -> total star dim = 32 + 9 = 41!
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

    player_rows = []
    for idx_in_order, q in enumerate(p_order):
        is_self = 1.0 if q == seat else 0.0
        st = stats[q]
        lvl = levels[q] / 10.0

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
            ],
            dtype=np.float32,
        )  # length 15 == N_BASE_PLAYER
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
        "star_x": star_x,                   # [S, 41]
        "edge_x": edge_x,                   # [S, S, 5]
        "player_x": player_x,               # [P, 15 + T + 10]
        "owner_rel": owner_rel,             # [S] in {-1, 0..P-1}
        "time_x": time_x,                   # [4]
        "own_mask": own_mask,               # [S]
        "non_own_mask": non_own_mask,       # [S]
        "free_own_mask": free_own_star,     # [S]
        "research_mask": research_mask,     # [1 + T]
        "spec_mask": spec_mask,             # [1 + MAX_SPECIALIST_ID]
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
        # Masked max pool per player
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
        # [B, S, S, H] -> [B, H, S, S]
        edge_bias = self.edge_proj(edge_x).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            h_s, h_p = layer(h_s, h_p, edge_bias, owner_rel)

        h_s = self.out_norm_s(h_s)
        h_p = self.out_norm_p(h_p)

        # Global representation pools self player + mean/max across ALL empires
        # so the policy & critic understand overall galaxy balance of power.
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

        # --- Group 1: Spatial Pointer Heads (index 0 = broad algorithmic prior) ---
        self.attack1_broad = mlp(g_dim, 1)
        self.attack1_star = mlp(d + g_dim, 1)
        self.attack2_broad = mlp(g_dim, 1)
        self.attack2_star = mlp(d + g_dim, 1)

        self.defend_broad = mlp(g_dim, 1)
        self.defend_star = mlp(d + g_dim, 1)

        self.ind_broad = mlp(g_dim, 1)
        self.ind_star = mlp(d, 1)
        self.sci_broad = mlp(g_dim, 1)
        self.sci_star = mlp(d, 1)
        self.wg_broad = mlp(g_dim, 1)
        self.wg_star = mlp(d, 1)

        with torch.no_grad():
            for head in (
                self.attack1_broad,
                self.attack2_broad,
                self.defend_broad,
                self.ind_broad,
                self.sci_broad,
                self.wg_broad,
            ):
                head[-1].bias.fill_(1.5)

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

        # --- Group 4: Discrete Macro Heads (Research, Specialist, Diplomacy) ---
        self.research_head = mlp(g_dim, 1 + n_techs)
        self.spec_head = mlp(g_dim, 1 + MAX_SPECIALIST_ID)
        self.spec_star_head = mlp(d, MAX_SPECIALIST_ID)
        with torch.no_grad():
            self.research_head[-1].bias[0] = KEEP_BIAS
            self.spec_head[-1].bias[0] = KEEP_BIAS

        # Auxiliary conservation-constrained hidden belief head:
        # [ships_logit, eco_logit, ind_logit, sci_logit] per star
        self.belief_head = mlp(d, 4)

        # Dual Distributional Critics:
        # - `fog_critic` evaluates fog observations (usable during search at test time!)
        # - `truth_critic` evaluates privileged full-galaxy observations during RL
        self.fog_critic = DistributionalValueHead(g_dim)
        self.truth_critic = DistributionalValueHead(g_dim)

    def _pointer_logits(
        self,
        broad_head: nn.Module,
        star_head: nn.Module,
        h_s: torch.Tensor,
        g: torch.Tensor,
        valid_mask: torch.Tensor,
        concat_global: bool = True,
    ) -> torch.Tensor:
        B, S, _ = h_s.shape
        broad = broad_head(g)  # [B, 1]
        if concat_global:
            inp = torch.cat([h_s, g.unsqueeze(1).expand(B, S, -1)], dim=-1)
        else:
            inp = h_s
        stars = star_head(inp).squeeze(-1)  # [B, S]
        stars = stars.masked_fill(~valid_mask, -1e9)
        return torch.cat([broad, stars], dim=-1)  # [B, 1 + S]

    def forward_policy(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        star_x = batch["star_x"]
        edge_x = batch["edge_x"]
        player_x = batch["player_x"]
        owner_rel = batch["owner_rel"]
        time_x = batch["time_x"]
        own_mask = batch["own_mask"]
        non_own_mask = batch["non_own_mask"]
        free_own_mask = batch["free_own_mask"]
        research_mask = batch["research_mask"]
        spec_mask = batch["spec_mask"]

        B, S, _ = star_x.shape
        P = player_x.shape[1]

        h_s, h_p, g = self.backbone(star_x, edge_x, player_x, owner_rel, time_x)

        att1_logits = self._pointer_logits(
            self.attack1_broad, self.attack1_star, h_s, g, non_own_mask, True
        )
        att2_logits = self._pointer_logits(
            self.attack2_broad, self.attack2_star, h_s, g, non_own_mask, True
        )
        def_logits = self._pointer_logits(
            self.defend_broad, self.defend_star, h_s, g, own_mask, True
        )
        ind_logits = self._pointer_logits(
            self.ind_broad, self.ind_star, h_s, g, own_mask, False
        )
        sci_logits = self._pointer_logits(
            self.sci_broad, self.sci_star, h_s, g, own_mask, False
        )
        wg_logits = self._pointer_logits(
            self.wg_broad, self.wg_star, h_s, g, own_mask, False
        )

        # Dirichlet concentrations > 0.2; starts right at DEFAULT_BUDGET_ALPHA * 3
        budget_raw = self.budget_head(g)
        budget_alpha = F.softplus(budget_raw + torch.log(torch.expm1(self.default_alpha * 3.0))) + 0.2

        knob_mean = torch.tanh(self.knob_head(g)) * 2.0
        knob_std = self.knob_log_std.clamp(-3.0, 0.5).exp().expand_as(knob_mean)

        res_logits = self.research_head(g).masked_fill(~research_mask, -1e9)

        # Specialist logits + domain prior
        sp_logits = self.spec_head(g)
        sp_prior = torch.zeros_like(sp_logits)
        for sid, (boost, _) in SPECIALIST_PRIOR.items():
            if sid <= MAX_SPECIALIST_ID:
                sp_prior[:, sid] = boost
        sp_logits = (sp_logits + sp_prior).masked_fill(~spec_mask, -1e9)

        sp_star_logits = self.spec_star_head(h_s)  # [B, S, MAX_SPEC]
        # Prefer stars with strong matching infra
        infra_feats = star_x[:, :, 11:14]  # eco, ind, sci log1p features
        sp_star_prior = torch.zeros_like(sp_star_logits)
        for sid, (_, ch) in SPECIALIST_PRIOR.items():
            if sid <= MAX_SPECIALIST_ID:
                sp_star_prior[:, :, sid - 1] = 2.5 * infra_feats[:, :, ch]
        has_any_free = free_own_mask.any(dim=1, keepdim=True)
        eff_free = torch.where(has_any_free, free_own_mask, own_mask)
        # Guard rare edge case when seat has 0 stars
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
            "att1_logits": att1_logits,
            "att2_logits": att2_logits,
            "def_logits": def_logits,
            "ind_logits": ind_logits,
            "sci_logits": sci_logits,
            "wg_logits": wg_logits,
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
        """Sample actions (when `actions is None` and `sample=True`), pick
        greedy actions (`sample=False`), or evaluate exact factored per-group
        `logp` & `entropy` of recorded `actions` for PPO."""
        out = self.forward_policy(batch)
        temp = max(float(temperature), 1e-3)

        pointer_keys = (
            ("att1", out["att1_logits"]),
            ("att2", out["att2_logits"]),
            ("def", out["def_logits"]),
            ("ind", out["ind_logits"]),
            ("sci", out["sci_logits"]),
            ("wg", out["wg_logits"]),
        )

        chosen: dict[str, torch.Tensor] = {} if actions is None else dict(actions)
        logp_spatial = torch.zeros(batch["star_x"].shape[0], device=batch["star_x"].device)
        ent_spatial = torch.zeros_like(logp_spatial)

        for name, logits in pointer_keys:
            scaled = logits / temp if (actions is None and sample) else logits
            dist = Categorical(logits=scaled)
            if actions is None:
                a = dist.sample() if sample else logits.argmax(dim=-1)
                chosen[name] = a
            else:
                a = actions[name]
            # Evaluate logp under true unscaled policy distribution (or temp if desired)
            eval_dist = Categorical(logits=logits)
            logp_spatial = logp_spatial + eval_dist.log_prob(a)
            ent_spatial = ent_spatial + eval_dist.entropy()

        # Budget Dirichlet [B, 5]
        alpha = out["budget_alpha"]
        dir_dist = Dirichlet(alpha)
        if actions is None:
            if sample:
                b = dir_dist.rsample()
            else:
                b = alpha / alpha.sum(dim=-1, keepdim=True)
            b = b.clamp(min=1e-4)
            b = b / b.sum(dim=-1, keepdim=True)
            chosen["budget"] = b
        else:
            b = actions["budget"].clamp(min=1e-4)
            b = b / b.sum(dim=-1, keepdim=True)
        logp_budget = dir_dist.log_prob(b)
        ent_budget = dir_dist.entropy()

        # Tactical continuous knobs [B, 3]
        knob_dist = Normal(out["knob_mean"], out["knob_std"])
        if actions is None:
            k_act = knob_dist.rsample() if sample else out["knob_mean"]
            chosen["knobs"] = k_act
        else:
            k_act = actions["knobs"]
        logp_knobs = knob_dist.log_prob(k_act).sum(dim=-1)
        ent_knobs = knob_dist.entropy().sum(dim=-1)

        # Macro discrete: research & specialist + specialist star
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

        # Gather star logits for selected specialist (when s_act > 0)
        s_idx = (s_act.clamp(min=1) - 1).view(-1, 1, 1).expand(
            -1, out["sp_star_logits"].shape[1], 1
        )
        sel_star_logits = out["sp_star_logits"].gather(2, s_idx).squeeze(-1)  # [B, S]
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

        hired_mask = (s_act > 0).to(out["res_logits"].dtype)
        logp_macro = (
            res_dist.log_prob(r_act)
            + sp_dist.log_prob(s_act)
            + hired_mask * sp_star_dist.log_prob(st_act)
        )
        ent_macro = (
            res_dist.entropy()
            + sp_dist.entropy()
            + hired_mask * sp_star_dist.entropy()
        )

        logp_groups = {
            "spatial": logp_spatial,
            "budget": logp_budget,
            "knobs": logp_knobs,
            "macro": logp_macro,
        }
        ent_groups = {
            "spatial": ent_spatial,
            "budget": ent_budget,
            "knobs": ent_knobs,
            "macro": ent_macro,
        }
        total_logp = logp_spatial + logp_budget + logp_knobs + logp_macro
        total_ent = ent_spatial + ent_budget + ent_knobs + ent_macro

        return {
            "actions": chosen,
            "logp": total_logp,
            "logp_groups": logp_groups,
            "entropy_groups": ent_groups,
            "entropy": total_ent,
            "policy_out": out,
        }
