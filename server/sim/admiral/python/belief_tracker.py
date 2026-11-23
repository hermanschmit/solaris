"""Conservation-constrained & scoreboard-delta belief tracker for Admiral.

Solves two structural blind spots in standard fog-of-war belief models:
1. Scoreboard-Conservation Projection: every player's total `[stars, ships,
   carriers, economy, industry, science]` is public every tick in Solaris.
   Instead of unconstrained per-star regression (which routinely hallucinates
   2x or 0.3x an opponent's true fleet), hidden-star ship and infrastructure
   estimates are projected onto each opponent's exact unseen remainder
   `PublicTotal(q) - KnownVisible(q)`.
2. Rolling Public Delta (`d_stats`) & Combat Attrition Detection: tracks
   public scoreboard trajectories over short/medium/cycle windows and compares
   actual `Delta ships` against expected manufacturing output from public
   `industry` and `manufacturing` tech, exposing off-screen wars and fleet
   depletion immediately even under total fog.
"""

import os
import sys
from collections import deque

import numpy as np

# Reuse existing read-only simulator helpers without modifying them.
_UPSTREAM_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python")
)
if _UPSTREAM_PY not in sys.path:
    sys.path.insert(0, _UPSTREAM_PY)

import belief_data as bd  # noqa: E402

HIDDEN = bd.HIDDEN
VISIBLE = bd.VISIBLE
VISIBLE_NO_SHIPS = bd.VISIBLE_NO_SHIPS

# Per-player dynamic delta features produced by BeliefTracker.delta_features:
#   0: d_stars_cycle
#   1: d_ships_short_rel      (4-tick relative ship change)
#   2: d_ships_cycle_rel      (1-cycle relative ship change)
#   3: attrition_short_log    (log1p unexplained ship loss over ~4 ticks)
#   4: attrition_cycle_log    (log1p unexplained ship loss over ~1 cycle)
#   5: d_carriers_cycle
#   6: d_eco_cycle
#   7: d_ind_cycle
#   8: d_sci_cycle
#   9: unseen_ship_fraction   (unseen ships / max(1, total ships))
N_DELTA_FEATURES = 10


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    z = x - np.max(x)
    e = np.exp(np.clip(z, -40.0, 40.0))
    s = float(e.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(x, 1.0 / len(x), dtype=np.float32)
    return (e / s).astype(np.float32)


def _integer_apportion(weights: np.ndarray, total: int) -> np.ndarray:
    """Hamilton (largest-remainder) apportionment: allocates non-negative
    integers summing to `total` proportionally to `weights`."""
    n = len(weights)
    if n == 0 or total <= 0:
        return np.zeros(n, dtype=np.int32)
    probs = weights / max(float(weights.sum()), 1e-8)
    exact = probs * float(total)
    floors = np.floor(exact).astype(np.int32)
    rem = int(total - int(floors.sum()))
    if rem > 0:
        frac = exact - floors
        order = np.argsort(-frac)[:rem]
        floors[order] += 1
    return floors


class BeliefTracker:
    """Per-seat stateful tracker maintained across a game."""

    def __init__(self, static: bd.Static, seat: int):
        self.static = static
        self.seat = int(seat)
        self.S = static.S
        self.P = static.P
        self.cycle = max(1, int(static.cycle))
        self._mfg_idx = (
            static.techs.index("manufacturing")
            if "manufacturing" in static.techs
            else 0
        )

        # Rolling history of (tick, stats[P, 6], levels[P, T])
        self.history: deque[tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=64
        )
        # Ghost incoming enemy ships remembered when visible carriers head to a
        # hidden star and leave scan range: star_idx -> [(arrival_tick, ships, owner)]
        self._ghosts: list[tuple[int, float, int, int]] = []
        self._last_tick = -1

        # Static natural resource richness score per star (0 for nebulas).
        self._nat_eco = np.zeros(self.S, np.float32)
        self._nat_ind = np.zeros(self.S, np.float32)
        self._nat_sci = np.zeros(self.S, np.float32)
        for i, s in enumerate(static.header["stars"]):
            if s.get("isNebula"):
                self._nat_eco[i] = 25.0
                self._nat_ind[i] = 25.0
                self._nat_sci[i] = 25.0
            else:
                nr = s.get("naturalResources") or {}
                if not isinstance(nr, dict):
                    nr = {"economy": nr, "industry": nr, "science": nr}
                self._nat_eco[i] = float(nr.get("economy") or 10.0)
                self._nat_ind[i] = float(nr.get("industry") or 10.0)
                self._nat_sci[i] = float(nr.get("science") or 10.0)

    def reset(self):
        self.history.clear()
        self._ghosts.clear()
        self._last_tick = -1

    def _expected_ship_production(
        self,
        stats_a: np.ndarray,
        levels_a: np.ndarray,
        stats_b: np.ndarray,
        levels_b: np.ndarray,
        dt: int,
    ) -> np.ndarray:
        """Expected ships manufactured by each player over `dt` ticks from
        public industry and manufacturing tech (`ind * (mfg + 5) / cycle`)."""
        if dt <= 0:
            return np.zeros(self.P, dtype=np.float32)
        avg_ind = 0.5 * (stats_a[:, 4] + stats_b[:, 4])
        avg_mfg = 0.5 * (
            levels_a[:, self._mfg_idx] + levels_b[:, self._mfg_idx]
        )
        per_tick = avg_ind * (avg_mfg + 5.0) / float(self.cycle)
        return (per_tick * float(dt)).astype(np.float32)

    def _snapshot_at_or_before(self, target_tick: int):
        if not self.history:
            return None
        chosen = self.history[0]
        for item in self.history:
            if item[0] <= target_tick:
                chosen = item
            else:
                break
        return chosen

    def update(self, frame: dict):
        """Ingest a new `seatFrame` observation (idempotent on same tick)."""
        tick = int(frame["tick"])
        stats = np.asarray(frame["stats"], dtype=np.float32)
        levels = np.asarray(frame["levels"], dtype=np.float32)

        if tick != self._last_tick:
            self.history.append((tick, stats.copy(), levels.copy()))
            self._last_tick = tick

            # Prune expired ghost arrivals (> 1 cycle past ETA) and record
            # newly seen enemy carriers headed to stars.
            self._ghosts = [
                g for g in self._ghosts if tick <= g[0] + self.cycle
            ]
            car = frame["carriers"]
            c_owner = np.asarray(car["owner"], dtype=np.int32)
            c_ships = np.asarray(car["ships"], dtype=np.float32)
            c_dst = np.asarray(car["destination"], dtype=np.int32)
            c_x = np.asarray(car["x"], dtype=np.float32)
            c_y = np.asarray(car["y"], dtype=np.float32)

            for idx in range(len(c_owner)):
                q = int(c_owner[idx])
                dst = int(c_dst[idx])
                sh = float(c_ships[idx])
                if q == self.seat or dst < 0 or sh <= 0:
                    continue
                dx = float(self.static.xy[dst, 0] - c_x[idx])
                dy = float(self.static.xy[dst, 1] - c_y[idx])
                eta = max(
                    1,
                    int(
                        np.ceil(
                            np.hypot(dx, dy) / max(self.static.speed, 1e-4)
                        )
                    ),
                )
                arr_tick = tick + eta
                # Deduplicate near-identical (owner, dst, arr_tick) entries.
                if not any(
                    g[2] == q and g[3] == dst and abs(g[0] - arr_tick) <= 1
                    for g in self._ghosts
                ):
                    self._ghosts.append((arr_tick, sh, q, dst))

    def _visible_accounting(self, frame: dict):
        """Compute per-player visible ship & infrastructure sums and unseen
        remainders from `frame`."""
        owner = np.asarray(frame["owner"], dtype=np.int32)
        vis = np.asarray(frame["visibility"], dtype=np.int32)
        mem_ships = np.asarray(frame["mem_ships"], dtype=np.float32)
        mem_infra = np.asarray(frame["mem_infra"], dtype=np.float32)  # [S, 3]
        stats = np.asarray(frame["stats"], dtype=np.float32)  # [P, 6]

        car = frame["carriers"]
        c_owner = np.asarray(car["owner"], dtype=np.int32)
        c_ships = np.asarray(car["ships"], dtype=np.float32)

        vis_ships = np.zeros(self.P, dtype=np.float32)
        vis_carriers = np.zeros(self.P, dtype=np.int32)
        vis_infra = np.zeros((self.P, 3), dtype=np.float32)

        star_ship_known = vis == VISIBLE
        star_infra_known = vis != HIDDEN

        for q in range(self.P):
            m_ship = (owner == q) & star_ship_known
            if m_ship.any():
                vis_ships[q] += float(mem_ships[m_ship].sum())
            m_car = c_owner == q
            if m_car.any():
                vis_carriers[q] = int(m_car.sum())
                known_c = m_car & (c_ships >= 0)
                if known_c.any():
                    vis_ships[q] += float(c_ships[known_c].sum())
            m_inf = (owner == q) & star_infra_known
            if m_inf.any():
                vis_infra[q] = mem_infra[m_inf].sum(axis=0)

        rem_ships = np.maximum(0.0, stats[:, 1] - vis_ships)
        rem_carriers = np.maximum(0, stats[:, 2].astype(np.int32) - vis_carriers)
        rem_infra = np.maximum(0.0, stats[:, 3:6] - vis_infra)
        return rem_ships, rem_carriers, rem_infra

    def delta_features(self, frame: dict) -> np.ndarray:
        """Return per-player dynamic scoreboard-delta & attrition features
        in canonical player index order `[P, N_DELTA_FEATURES]`."""
        self.update(frame)
        tick = int(frame["tick"])
        stats = np.asarray(frame["stats"], dtype=np.float32)
        levels = np.asarray(frame["levels"], dtype=np.float32)
        rem_ships, _, _ = self._visible_accounting(frame)

        out = np.zeros((self.P, N_DELTA_FEATURES), dtype=np.float32)

        snap_short = self._snapshot_at_or_before(max(0, tick - 4))
        snap_cycle = self._snapshot_at_or_before(max(0, tick - self.cycle))

        if snap_short is not None:
            t0, s0, l0 = snap_short
            dt = max(1, tick - t0)
            d_ships = stats[:, 1] - s0[:, 1]
            denom = np.maximum(20.0, s0[:, 1])
            out[:, 1] = np.clip(d_ships / denom, -2.0, 2.0)
            expected = self._expected_ship_production(s0, l0, stats, levels, dt)
            attrition = np.maximum(0.0, expected - d_ships)
            out[:, 3] = np.log1p(attrition) / 5.0

        if snap_cycle is not None:
            t0, s0, l0 = snap_cycle
            dt = max(1, tick - t0)
            scale = float(self.cycle) / float(dt)
            out[:, 0] = np.clip((stats[:, 0] - s0[:, 0]) * scale / 10.0, -2.0, 2.0)
            d_ships = stats[:, 1] - s0[:, 1]
            denom = np.maximum(20.0, s0[:, 1])
            out[:, 2] = np.clip((d_ships / denom) * min(scale, 3.0), -2.0, 2.0)
            expected = self._expected_ship_production(s0, l0, stats, levels, dt)
            attrition = np.maximum(0.0, expected - d_ships)
            out[:, 4] = np.log1p(attrition) / 6.0
            out[:, 5] = np.clip((stats[:, 2] - s0[:, 2]) * scale / 5.0, -2.0, 2.0)
            out[:, 6] = np.clip((stats[:, 3] - s0[:, 3]) * scale / 10.0, -2.0, 2.0)
            out[:, 7] = np.clip((stats[:, 4] - s0[:, 4]) * scale / 10.0, -2.0, 2.0)
            out[:, 8] = np.clip((stats[:, 5] - s0[:, 5]) * scale / 5.0, -2.0, 2.0)

        out[:, 9] = rem_ships / np.maximum(1.0, stats[:, 1])
        out[self.seat, 9] = 0.0
        return out

    def _prior_star_logits(self, frame: dict) -> dict[str, np.ndarray]:
        """Physics-informed prior log-weights for hidden stars' ships and
        infrastructure before neural residual adjustments."""
        tick = int(frame["tick"])
        owner = np.asarray(frame["owner"], dtype=np.int32)
        seen_tick = np.asarray(frame["seen_tick"], dtype=np.int32)
        mem_ships = np.asarray(frame["mem_ships"], dtype=np.float32)
        mem_infra = np.asarray(frame["mem_infra"], dtype=np.float32)
        levels = np.asarray(frame["levels"], dtype=np.float32)
        stats = np.asarray(frame["stats"], dtype=np.float32)

        own_mask = owner == self.seat
        if own_mask.any():
            dist_to_own = self.static.travel[:, own_mask].min(axis=1)
        else:
            dist_to_own = np.full(self.S, float(self.cycle * 4), dtype=np.float32)

        # Frontier stars close to us or enemy capitals tend to hold larger garrisons.
        frontier_boost = np.exp(-dist_to_own / max(float(self.cycle * 2), 1.0))
        home_boost = (self.static.home >= 0).astype(np.float32)

        ship_prior = np.zeros(self.S, dtype=np.float32)
        eco_prior = np.log1p(self._nat_eco) * 0.6 + home_boost * 0.9
        ind_prior = np.log1p(self._nat_ind) * 0.6 + home_boost * 1.0 + 0.3 * frontier_boost
        sci_prior = np.log1p(self._nat_sci) * 0.6 + home_boost * 0.8

        ghost_bonus = np.zeros(self.S, dtype=np.float32)
        for arr_tick, sh, q, dst in self._ghosts:
            if 0 <= dst < self.S and owner[dst] == q and tick >= arr_tick - 2:
                ghost_bonus[dst] += sh

        for i in range(self.S):
            q = int(owner[i])
            if q < 0 or q == self.seat:
                continue
            if seen_tick[i] >= 0:
                dt = max(0, tick - int(seen_tick[i]))
                mfg = float(levels[q, self._mfg_idx])
                ind = float(mem_infra[i, 1])
                grown = (
                    float(mem_ships[i])
                    + ind * (mfg + 5.0) * float(dt) / float(self.cycle)
                    + float(ghost_bonus[i])
                )
                ship_prior[i] = np.log1p(max(0.5, grown))
                eco_prior[i] = np.log1p(max(0.5, float(mem_infra[i, 0])))
                ind_prior[i] = np.log1p(max(0.5, float(mem_infra[i, 1])))
                sci_prior[i] = np.log1p(max(0.5, float(mem_infra[i, 2])))
            else:
                avg_ships = max(
                    4.0, float(stats[q, 1]) / max(1.0, float(stats[q, 0]))
                )
                shape_factor = (
                    0.55
                    + 0.70 * home_boost[i]
                    + 0.55 * frontier_boost[i]
                    + 0.01 * self._nat_ind[i]
                )
                est_unseen = avg_ships * shape_factor + float(ghost_bonus[i])
                ship_prior[i] = float(np.log1p(max(est_unseen, 0.5)))

        return {
            "ships": ship_prior,
            "economy": eco_prior.astype(np.float32),
            "industry": ind_prior.astype(np.float32),
            "science": sci_prior.astype(np.float32),
        }

    def constrain_beliefs(
        self,
        frame: dict,
        learned_ship_logits: np.ndarray | None = None,
        learned_ind_logits: np.ndarray | None = None,
        learned_eco_logits: np.ndarray | None = None,
        learned_sci_logits: np.ndarray | None = None,
        include_transit_share: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute conservation-constrained star beliefs `[S]` for `ships`,
        `economy`, `industry`, and `science` (+ `transit_ships [P]`).

        Guarantees that for every opponent `q`:
          sum_{s owned by q} believed_ships[s] + visible_carrier_ships[q]
          + transit_ships[q] == max(visible_known_ships[q], public_total_ships[q])
        (with `transit_ships[q] == 0` when `include_transit_share=False`).
        """
        self.update(frame)
        owner = np.asarray(frame["owner"], dtype=np.int32)
        vis = np.asarray(frame["visibility"], dtype=np.int32)
        mem_ships = np.asarray(frame["mem_ships"], dtype=np.float32)
        mem_infra = np.asarray(frame["mem_infra"], dtype=np.float32)

        rem_ships, rem_carriers, rem_infra = self._visible_accounting(frame)
        priors = self._prior_star_logits(frame)

        ship_logits = priors["ships"].copy()
        if learned_ship_logits is not None:
            ship_logits = ship_logits + np.asarray(
                learned_ship_logits, dtype=np.float32
            )
        ind_logits = priors["industry"].copy()
        if learned_ind_logits is not None:
            ind_logits = ind_logits + np.asarray(
                learned_ind_logits, dtype=np.float32
            )
        eco_logits = priors["economy"].copy()
        if learned_eco_logits is not None:
            eco_logits = eco_logits + np.asarray(
                learned_eco_logits, dtype=np.float32
            )
        sci_logits = priors["science"].copy()
        if learned_sci_logits is not None:
            sci_logits = sci_logits + np.asarray(
                learned_sci_logits, dtype=np.float32
            )

        ships = np.where(vis == VISIBLE, mem_ships, 0.0).astype(np.float32)
        eco = np.where(vis != HIDDEN, mem_infra[:, 0], 0.0).astype(np.float32)
        ind = np.where(vis != HIDDEN, mem_infra[:, 1], 0.0).astype(np.float32)
        sci = np.where(vis != HIDDEN, mem_infra[:, 2], 0.0).astype(np.float32)
        transit = np.zeros(self.P, dtype=np.float32)

        # Unowned hidden stars have 0 infrastructure and 0 garrison in standard Solaris.
        for q in range(self.P):
            if q == self.seat:
                continue
            h_ships = np.flatnonzero((owner == q) & (vis != VISIBLE))
            if len(h_ships) > 0 and rem_ships[q] > 0:
                if include_transit_share and rem_carriers[q] > 0:
                    # Virtual transit slot represents ships mid-flight on unseen carriers.
                    transit_logit = float(
                        np.mean(ship_logits[h_ships])
                        + np.log(0.35 * rem_carriers[q] / max(len(h_ships), 1) + 1e-3)
                    )
                    aug = np.append(ship_logits[h_ships], transit_logit)
                    w = _softmax(aug)
                    ships[h_ships] = w[:-1] * rem_ships[q]
                    transit[q] = float(w[-1] * rem_ships[q])
                else:
                    w = _softmax(ship_logits[h_ships])
                    ships[h_ships] = w * rem_ships[q]
            elif len(h_ships) == 0 and rem_ships[q] > 0:
                transit[q] = float(rem_ships[q])

            h_inf = np.flatnonzero((owner == q) & (vis == HIDDEN))
            if len(h_inf) > 0:
                if rem_infra[q, 0] > 0:
                    eco[h_inf] = _softmax(eco_logits[h_inf]) * rem_infra[q, 0]
                if rem_infra[q, 1] > 0:
                    ind[h_inf] = _softmax(ind_logits[h_inf]) * rem_infra[q, 1]
                if rem_infra[q, 2] > 0:
                    sci[h_inf] = _softmax(sci_logits[h_inf]) * rem_infra[q, 2]

        return {
            "ships": ships,
            "economy": eco,
            "industry": ind,
            "science": sci,
            "transit_ships": transit,
        }

    def executor_beliefs(
        self,
        frame: dict,
        learned_ship_logits: np.ndarray | None = None,
        learned_ind_logits: np.ndarray | None = None,
    ) -> dict[str, list[float | None]]:
        """Format conservation-constrained beliefs for `AdmiralDecision.beliefs`."""
        vis = np.asarray(frame["visibility"], dtype=np.int32)
        est = self.constrain_beliefs(
            frame,
            learned_ship_logits=learned_ship_logits,
            learned_ind_logits=learned_ind_logits,
            include_transit_share=False,
        )
        ships_out = [
            None if int(vis[i]) == VISIBLE else round(float(est["ships"][i]), 1)
            for i in range(self.S)
        ]
        ind_out = [
            None if int(vis[i]) != HIDDEN else round(float(est["industry"][i]), 1)
            for i in range(self.S)
        ]
        return {"ships": ships_out, "industry": ind_out}

    def sample_belief_worlds(
        self,
        frame: dict,
        k_worlds: int = 3,
        rng: np.random.Generator | None = None,
        learned_ship_logits: np.ndarray | None = None,
        learned_ind_logits: np.ndarray | None = None,
        noise_scale: float = 0.65,
    ) -> list[dict[str, list[int]]]:
        """Generate `k_worlds` integer conservation-exact `BeliefWorld` dicts
        for `AdmiralEnv.lookaheadAdmiral`. World 0 is always the deterministic
        MAP estimate; worlds `1..k_worlds-1` sample Gumbel-perturbed Hidden
        allocations that still sum to each opponent's exact scoreboard total."""
        if rng is None:
            rng = np.random.default_rng(1)
        self.update(frame)
        owner = np.asarray(frame["owner"], dtype=np.int32)
        vis = np.asarray(frame["visibility"], dtype=np.int32)
        mem_ships = np.asarray(frame["mem_ships"], dtype=np.float32)
        mem_infra = np.asarray(frame["mem_infra"], dtype=np.float32)

        rem_ships, _, rem_infra = self._visible_accounting(frame)
        priors = self._prior_star_logits(frame)

        base_ship = priors["ships"].copy()
        if learned_ship_logits is not None:
            base_ship = base_ship + np.asarray(learned_ship_logits, np.float32)
        base_ind = priors["industry"].copy()
        if learned_ind_logits is not None:
            base_ind = base_ind + np.asarray(learned_ind_logits, np.float32)
        base_eco = priors["economy"].copy()
        base_sci = priors["science"].copy()

        worlds: list[dict[str, list[int]]] = []
        for wi in range(max(1, int(k_worlds))):
            ships = np.where(vis == VISIBLE, np.rint(mem_ships), 0).astype(np.int32)
            eco = np.where(vis != HIDDEN, np.rint(mem_infra[:, 0]), 0).astype(np.int32)
            ind = np.where(vis != HIDDEN, np.rint(mem_infra[:, 1]), 0).astype(np.int32)
            sci = np.where(vis != HIDDEN, np.rint(mem_infra[:, 2]), 0).astype(np.int32)

            sigma = 0.0 if wi == 0 else float(noise_scale)
            for q in range(self.P):
                if q == self.seat:
                    continue
                h_s = np.flatnonzero((owner == q) & (vis != VISIBLE))
                if len(h_s) > 0 and rem_ships[q] > 0:
                    logits = base_ship[h_s]
                    if sigma > 0:
                        logits = logits + rng.gumbel(0.0, sigma, size=len(h_s)).astype(
                            np.float32
                        )
                    ships[h_s] = _integer_apportion(
                        _softmax(logits), int(round(float(rem_ships[q])))
                    )

                h_i = np.flatnonzero((owner == q) & (vis == HIDDEN))
                if len(h_i) > 0:
                    for arr, base_l, rem_val in (
                        (eco, base_eco, rem_infra[q, 0]),
                        (ind, base_ind, rem_infra[q, 1]),
                        (sci, base_sci, rem_infra[q, 2]),
                    ):
                        if rem_val > 0:
                            logits = base_l[h_i]
                            if sigma > 0:
                                logits = logits + rng.gumbel(
                                    0.0, sigma * 0.75, size=len(h_i)
                                ).astype(np.float32)
                            arr[h_i] = _integer_apportion(
                                _softmax(logits), int(round(float(rem_val)))
                            )

            worlds.append(
                {
                    "ships": ships.tolist(),
                    "economy": eco.tolist(),
                    "industry": ind.tolist(),
                    "science": sci.tolist(),
                }
            )
        return worlds
