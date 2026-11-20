"""`AdmiralAgent` and 1-Cycle Belief-Determinized Lookahead Planner (`AdmiralSearchPlanner`)."""

import math
import os
import sys

import numpy as np
import torch

_UPSTREAM_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python")
)
if _UPSTREAM_PY not in sys.path:
    sys.path.insert(0, _UPSTREAM_PY)

import belief_data as bd  # noqa: E402
from admiral_net import (  # noqa: E402
    TACTICAL_KNOBS,
    AdmiralNet,
    build_admiral_inputs,
)
from belief_tracker import BeliefTracker  # noqa: E402


def _to_torch_batch(np_inputs_list: list[dict[str, np.ndarray]], device) -> dict[str, torch.Tensor]:
    keys = np_inputs_list[0].keys()
    out = {}
    for k in keys:
        stacked = np.stack([item[k] for item in np_inputs_list], axis=0)
        t = torch.from_numpy(stacked).to(device)
        out[k] = t
    return out


def decode_admiral_decision(
    static: bd.Static,
    frame: dict,
    seat: int,
    actions: dict[str, np.ndarray | int | float],
    belief_logits: np.ndarray | None = None,
    tracker: BeliefTracker | None = None,
    send_beliefs: bool = True,
) -> dict:
    """Convert sampled/greedy `AdmiralNet` action outputs into an `AdmiralDecision`
    JSON dictionary for `TacticalExecutor`."""
    S = static.S
    cycle = max(1, int(static.cycle))
    owner = np.asarray(frame["owner"], dtype=np.int32)

    decision: dict = {}

    # 1. Budget split [science, industry, economy, warpGate, carrierReserve]
    budget = np.asarray(actions["budget"], dtype=np.float64)
    budget = np.maximum(budget, 1e-4)
    budget = (budget / budget.sum()).tolist()
    decision["budgetSplit"] = [round(float(x), 4) for x in budget]

    # 2. Continuous tactical knobs
    knobs = np.asarray(actions["knobs"], dtype=np.float64)
    for (name, default, scale, lo, hi), val in zip(TACTICAL_KNOBS, knobs):
        mapped = default * math.exp(scale * float(val))
        mapped = min(max(mapped, lo), hi)
        decision[name] = round(mapped, 3)

    # 3. Offensive spatial pointers -> smooth spearhead `attackDesire`
    att1 = int(actions["att1"])
    att2 = int(actions["att2"])
    if att1 > 0 or att2 > 0:
        desire = np.full(S, 0.55, dtype=np.float32)
        radius = max(4.0, 0.65 * float(cycle))
        for ptr, boost in ((att1, 3.5), (att2, 2.2)):
            if 1 <= ptr <= S:
                s_idx = ptr - 1
                dist_ticks = static.travel[s_idx]
                desire += boost * np.exp(-dist_ticks / radius)
        decision["attackDesire"] = [round(float(x), 3) for x in desire]

    # 4. Defensive bastion pointer -> proactive stronghold `garrisonTarget`
    def_ptr = int(actions["def"])
    if 1 <= def_ptr <= S:
        d_idx = def_ptr - 1
        radius = max(4.0, 0.5 * float(cycle))
        own_mask = owner == seat
        enemy_mask = (owner >= 0) & (owner != seat)
        garrison = np.zeros(S, dtype=np.float32)
        if own_mask.any():
            # Base border awareness + bastion concentration around d_idx
            for i in np.flatnonzero(own_mask):
                near_enemy = (
                    float(static.travel[i, enemy_mask].min())
                    if enemy_mask.any()
                    else 999.0
                )
                border_base = 12.0 if near_enemy <= float(cycle * 1.5) else 0.0
                bastion = 28.0 * math.exp(
                    -float(static.travel[d_idx, i]) / radius
                )
                garrison[i] = border_base + bastion
        decision["garrisonTarget"] = [round(float(x), 1) for x in garrison]

    # 5. Infrastructure & Warp-Gate pointers
    for ptr_key, field, peak in (
        ("ind", "industryPriority", 4.5),
        ("sci", "sciencePriority", 4.5),
        ("wg", "warpGatePriority", 6.0),
    ):
        ptr = int(actions[ptr_key])
        if 1 <= ptr <= S:
            u_idx = ptr - 1
            prio = np.ones(S, dtype=np.float32)
            prio[u_idx] = peak
            decision[field] = [round(float(x), 3) for x in prio]

    # 6. Research & Star Specialist
    r_act = int(actions["research"])
    if 1 <= r_act <= len(static.techs):
        decision["research"] = static.techs[r_act - 1]

    s_act = int(actions["spec"])
    st_act = int(actions["spec_star"])
    if s_act > 0 and 0 <= st_act < S:
        decision["hire"] = [{"star": st_act, "specialist": s_act}]

    # 7. Conservation-constrained hidden ships & industry beliefs
    if send_beliefs and tracker is not None:
        ship_l = belief_logits[:, 0] if belief_logits is not None else None
        ind_l = belief_logits[:, 2] if belief_logits is not None else None
        decision["beliefs"] = tracker.executor_beliefs(
            frame, learned_ship_logits=ship_l, learned_ind_logits=ind_l
        )

    return decision


class AdmiralAgent:
    """Stateful per-game wrapper managing `BeliefTracker`s across seats and
    producing `AdmiralDecision`s (+ optional rollout training records)."""

    def __init__(
        self,
        model: AdmiralNet,
        static: bd.Static,
        sample: bool = False,
        temperature: float = 1.0,
        send_beliefs: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.static = static
        self.sample = bool(sample)
        self.temperature = float(temperature)
        self.send_beliefs = bool(send_beliefs)
        self.device = torch.device(device)
        self.trackers: dict[int, BeliefTracker] = {}

    def tracker_for(self, seat: int) -> BeliefTracker:
        seat = int(seat)
        if seat not in self.trackers:
            self.trackers[seat] = BeliefTracker(self.static, seat)
        return self.trackers[seat]

    @torch.no_grad()
    def decide(
        self, obs: dict, seat: int, record_step: bool = False
    ) -> tuple[dict, dict | None]:
        seat = int(seat)
        tracker = self.tracker_for(seat)
        frame = obs["frame"]
        own_facts = obs.get("ownFacts")
        ticks_to_end = obs.get("ticksToEnd")

        np_in = build_admiral_inputs(
            self.static,
            frame,
            seat,
            own_facts=own_facts,
            tracker=tracker,
            ticks_to_end=ticks_to_end,
        )
        batch = _to_torch_batch([np_in], self.device)
        res = self.model.act_or_evaluate(
            batch, sample=self.sample, temperature=self.temperature
        )

        act_np = {
            k: v[0].detach().cpu().numpy() for k, v in res["actions"].items()
        }
        belief_np = (
            res["policy_out"]["belief_logits"][0].detach().cpu().numpy()
        )

        decision = decode_admiral_decision(
            self.static,
            frame,
            seat,
            act_np,
            belief_logits=belief_np,
            tracker=tracker,
            send_beliefs=self.send_beliefs,
        )

        if not record_step:
            return decision, None

        rec = {
            "inputs": np_in,
            "act": act_np,
            "logp": float(res["logp"][0].item()),
            "logp_groups": {
                k: float(v[0].item()) for k, v in res["logp_groups"].items()
            },
        }
        return decision, rec


class AdmiralSearchPlanner:
    """1-Cycle Fog-Legal Belief-Determinized Lookahead Planner.

    At cycle boundaries, generates `n_plans` candidate `AdmiralDecision`s
    (Plan 0 = greedy policy mode; Plans `1..n_plans-1` = diverse policy samples),
    samples `n_worlds` conservation-exact `BeliefWorld`s from `BeliefTracker`,
    simulates every `(plan, world)` pair forward for 1 production cycle via
    `env.lookahead_admiral`, and selects the plan with highest expected
    `fog_value` + empire-share gain across the `n_worlds` determinizations.
    """

    def __init__(
        self,
        agent: AdmiralAgent,
        n_plans: int = 4,
        n_worlds: int = 2,
        lookahead_ticks: int | None = None,
        sample_temperature: float = 1.15,
    ):
        self.agent = agent
        self.n_plans = max(2, int(n_plans))
        self.n_worlds = max(1, int(n_worlds))
        self.lookahead_ticks = lookahead_ticks
        self.sample_temperature = float(sample_temperature)

    @torch.no_grad()
    def decide(self, env, obs: dict, seat: int, seed: int = 1) -> dict:
        seat = int(seat)
        static = self.agent.static
        cycle = max(1, int(static.cycle))
        tick = int(obs["tick"])

        # Only spend search compute at cycle boundaries (when macro upgrades &
        # strategy shifts happen); mid-cycle reactive wake-ups use fast policy inference.
        if tick % cycle != 1:
            return self.agent.decide(obs, seat, record_step=False)[0]

        tracker = self.agent.tracker_for(seat)
        frame = obs["frame"]
        own_facts = obs.get("ownFacts")
        ticks_to_end = obs.get("ticksToEnd")

        np_in = build_admiral_inputs(
            static,
            frame,
            seat,
            own_facts=own_facts,
            tracker=tracker,
            ticks_to_end=ticks_to_end,
        )
        batch_1 = _to_torch_batch([np_in], self.agent.device)

        # Candidate 0: greedy policy mode
        greedy_res = self.agent.model.act_or_evaluate(batch_1, sample=False)
        belief_np = (
            greedy_res["policy_out"]["belief_logits"][0].detach().cpu().numpy()
        )
        candidates_act = [
            {k: v[0].detach().cpu().numpy() for k, v in greedy_res["actions"].items()}
        ]

        # Candidates 1..M-1: batched diverse samples from the policy
        n_extra = self.n_plans - 1
        if n_extra > 0:
            batch_m = _to_torch_batch([np_in] * n_extra, self.agent.device)
            samp_res = self.agent.model.act_or_evaluate(
                batch_m, sample=True, temperature=self.sample_temperature
            )
            for m in range(n_extra):
                candidates_act.append(
                    {
                        k: v[m].detach().cpu().numpy()
                        for k, v in samp_res["actions"].items()
                    }
                )

        plans = [
            decode_admiral_decision(
                static,
                frame,
                seat,
                act_dict,
                belief_logits=belief_np,
                tracker=tracker,
                send_beliefs=self.agent.send_beliefs,
            )
            for act_dict in candidates_act
        ]

        rng = np.random.default_rng(int(seed) * 1009 + tick * 37 + seat)
        worlds = tracker.sample_belief_worlds(
            frame,
            k_worlds=self.n_worlds,
            rng=rng,
            learned_ship_logits=belief_np[:, 0],
            learned_ind_logits=belief_np[:, 2],
        )

        horizon = (
            int(self.lookahead_ticks)
            if self.lookahead_ticks is not None
            else cycle
        )
        rollouts = env.lookahead_admiral(
            seat, plans, worlds, ticks=horizon, seed=int(seed) + tick
        )

        # Batch-evaluate all M * K leaf observations with `fog_value`
        leaf_inputs = []
        leaf_share = []
        for m in range(len(plans)):
            for wi in range(len(worlds)):
                leaf = rollouts[m][wi]
                lf_frame = leaf["frame"]
                lf_own = leaf["ownFacts"]
                rem_end = (
                    max(0, int(ticks_to_end) - horizon)
                    if ticks_to_end is not None
                    else None
                )
                leaf_inputs.append(
                    build_admiral_inputs(
                        static,
                        lf_frame,
                        seat,
                        own_facts=lf_own,
                        tracker=None,
                        ticks_to_end=rem_end,
                    )
                )
                players = leaf["players"]
                tot_stars = max(1, sum(p["stars"] for p in players))
                tot_ships = max(1, sum(p["ships"] for p in players))
                me = players[seat]
                share = (
                    0.6 * (me["stars"] / tot_stars)
                    + 0.4 * (me["ships"] / tot_ships)
                    - (1.0 if me["defeated"] else 0.0)
                )
                leaf_share.append(share)

        leaf_batch = _to_torch_batch(leaf_inputs, self.agent.device)
        leaf_out = self.agent.model.forward_policy(leaf_batch)
        fog_vals = leaf_out["fog_value"].detach().cpu().numpy()
        combined = fog_vals + 0.25 * np.asarray(leaf_share, dtype=np.float32)
        scores = combined.reshape(len(plans), len(worlds)).mean(axis=1)

        best_idx = int(np.argmax(scores))
        return plans[best_idx]
