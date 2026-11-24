"""Persistent JSON-lines RPC worker for Live Diplomatic Admiral.

Decoupled 100% from `AdmiralNet` checkpoint internals: imports `AdmiralNet`,
`build_admiral_inputs`, `decode_admiral_decision`, and `BeliefTracker` directly
from `server/sim/admiral/python/` without modifying any training file, so
upcoming `ema_*.pt` / `ckpt_*.pt` checkpoints drop in seamlessly.

Enforces Invariant 5 (Zero-Shot Neural Adaptation for Diplomatic Borders):
1. Offensive frontier masking: `frontier_mask` allows ONLY unowned (uncontested)
   and `HOSTILE` stars — `ALLIED` and `PROTECTED` stars are hard-masked out.
2. Target empire masking: `alive_enemy_mask` masks out `ALLIED` and `PROTECTED`
   empires (preserving index 0 = opportunistic / no rival when at peace).
3. Border feature adaptation: `star_x` zeroes `enemy` threat on `ALLIED` stars,
   down-weights `PROTECTED` deterrence borders, and focuses `ticks_to_enemy` /
   `in_enemy_jump` on `HOSTILE` war fronts so garrisons & logistics flow to
   active enemies.
4. Research & specialist masks: applies `ResearchCoordinator` eligibility masks
   and `specialistBans`, returning full `resLogits` for `#1`/`#2` queue ranking.
5. `BeliefTracker` persistence & Time-Machine backfill warmup (`restore_tracker`,
   `ingest_frames`, `export_tracker`).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ADMIRAL_PY = _THIS_DIR.parent.parent / "python"
_UPSTREAM_PY = _THIS_DIR.parents[2] / "python"

for p in (str(_ADMIRAL_PY), str(_UPSTREAM_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import belief_data as bd  # noqa: E402
from admiral_agent import _to_torch_batch, decode_admiral_decision  # noqa: E402
from admiral_net import (  # noqa: E402
    MAX_SPECIALIST_ID,
    AdmiralNet,
    build_admiral_inputs,
)
from belief_tracker import BeliefTracker  # noqa: E402


def _load_compatible_checkpoint(
    n_techs: int, checkpoint_path: str | None = None
) -> AdmiralNet:
    if not checkpoint_path:
        model = AdmiralNet(n_techs)
        model.eval()
        return model

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved.get("config", {}) if isinstance(saved, dict) else {}
    state = (
        saved["model"]
        if isinstance(saved, dict) and "model" in saved
        else saved
    )

    # Infer `d` and `n_layers` from `state` if `config` was omitted in a raw state_dict!
    d = int(cfg.get("d", 0))
    if d <= 0 and "backbone.star_proj.weight" in state:
        d = int(state["backbone.star_proj.weight"].shape[0])
    if d <= 0:
        d = 128

    n_layers = int(cfg.get("n_layers", 0))
    if n_layers <= 0:
        layer_idxs = [
            int(k.split(".")[2])
            for k in state.keys()
            if k.startswith("backbone.layers.")
        ]
        n_layers = (max(layer_idxs) + 1) if layer_idxs else 4

    n_heads = int(cfg.get("n_heads", 4))
    ckpt_techs = int(cfg.get("n_techs", n_techs))

    model = AdmiralNet(
        n_techs=ckpt_techs, d=d, n_layers=n_layers, n_heads=n_heads
    )
    # Support both full PPO/EMA checkpoints (`strict=True`) and Stage-1 `.sbd`
    # pretrain checkpoints (`strict=False` if policy heads were omitted).
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def serialize_tracker(tracker: BeliefTracker) -> dict:
    return {
        "seat": int(tracker.seat),
        "lastTick": int(tracker._last_tick),
        "history": [
            [int(t), st.tolist(), lv.tolist()]
            for (t, st, lv) in tracker.history
        ],
        "ghosts": [
            [int(arr_t), float(sh), int(q), int(dst)]
            for (arr_t, sh, q, dst) in tracker._ghosts
        ],
    }


def restore_tracker(tracker: BeliefTracker, saved: dict | None):
    if not saved:
        return
    tracker.reset()
    tracker._last_tick = int(saved.get("lastTick", -1))
    for item in saved.get("history", []):
        t, st, lv = item
        tracker.history.append(
            (
                int(t),
                np.asarray(st, dtype=np.float32),
                np.asarray(lv, dtype=np.float32),
            )
        )
    for g in saved.get("ghosts", []):
        arr_t, sh, q, dst = g
        tracker._ghosts.append((int(arr_t), float(sh), int(q), int(dst)))


def apply_diplomatic_adaptation(
    np_in: dict[str, np.ndarray],
    static: bd.Static,
    frame: dict,
    seat: int,
    diplomatic_tiers: list[str] | None,
    neutral_border_posture: str = "defend",
    research_allowed_mask: list[bool] | None = None,
    banned_star_specialists: list[int] | None = None,
    contested_unowned_stars: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Mutates `np_in` in-place to enforce Invariant 5 diplomatic masks & features."""
    S, P = static.S, static.P
    cycle = max(1, int(static.cycle))
    max_t = float(cycle * 6)
    owner = np.asarray(frame["owner"], dtype=np.int64)
    levels = np.asarray(frame["levels"], dtype=np.float32)

    if diplomatic_tiers and len(diplomatic_tiers) == P:
        tiers = list(diplomatic_tiers)
        tiers[seat] = "ALLIED"

        hostile_pids = {q for q, tr in enumerate(tiers) if tr == "HOSTILE"}
        allied_pids = {q for q, tr in enumerate(tiers) if q != seat and tr == "ALLIED"}
        protected_pids = {q for q, tr in enumerate(tiers) if tr == "PROTECTED"}

        is_hostile_star = np.isin(owner, list(hostile_pids)) if hostile_pids else np.zeros(S, dtype=bool)
        is_allied_star = np.isin(owner, list(allied_pids)) if allied_pids else np.zeros(S, dtype=bool)
        is_protected_star = np.isin(owner, list(protected_pids)) if protected_pids else np.zeros(S, dtype=bool)
        is_unowned = owner < 0

        contested_mask = np.zeros(S, dtype=bool)
        for c_idx in contested_unowned_stars or []:
            if 0 <= int(c_idx) < S:
                contested_mask[int(c_idx)] = True

        # 1. Frontier mask: ONLY unowned (uncontested) or HOSTILE stars!
        allowed_frontier = (is_unowned & ~contested_mask) | is_hostile_star
        np_in["frontier_mask"] = np_in["frontier_mask"] & allowed_frontier

        # 2. Target player mask (`p_order = [seat] + [q != seat]`): ONLY HOSTILE players (+ index 0)!
        p_order = [seat] + [q for q in range(P) if q != seat]
        enemy_mask = np_in["alive_enemy_mask"].copy()
        enemy_mask[0] = True
        for idx_in_order, q in enumerate(p_order):
            if idx_in_order >= 1 and tiers[q] != "HOSTILE":
                enemy_mask[idx_in_order] = False
        np_in["alive_enemy_mask"] = enemy_mask

        # 3. Star tensor (`star_x`) border adaptation:
        star_x = np_in["star_x"]
        # Col 3 = enemy indicator: 1.0 hostile, 0.25 protected deterrence (0.0 if open), 0.0 allied
        prot_val = 0.25 if neutral_border_posture != "open" else 0.0
        enemy_col = np.zeros(S, dtype=np.float32)
        enemy_col[is_hostile_star] = 1.0
        enemy_col[is_protected_star] = prot_val
        star_x[:, 3] = enemy_col

        # Col 18 & 19 = inbound enemy ships & ETA: exclude ALLIED carriers!
        if allied_pids:
            car = frame["carriers"]
            c_owner = np.asarray(car["owner"], dtype=np.int32)
            c_ships = np.maximum(np.asarray(car["ships"], dtype=np.float32), 0.0)
            c_dst = np.asarray(car["destination"], dtype=np.int32)
            c_x = np.asarray(car["x"], dtype=np.float32)
            c_y = np.asarray(car["y"], dtype=np.float32)
            inbound = np.zeros(S, dtype=np.float32)
            min_eta = np.full(S, float(cycle * 3), dtype=np.float32)
            for idx in range(len(c_owner)):
                q = int(c_owner[idx])
                dst = int(c_dst[idx])
                if q == seat or q in allied_pids or dst < 0 or dst >= S:
                    continue
                inbound[dst] += max(float(c_ships[idx]), 1.0)
                eta = np.hypot(
                    float(static.xy[dst, 0] - c_x[idx]),
                    float(static.xy[dst, 1] - c_y[idx]),
                ) / max(static.speed, 1e-4)
                if eta < min_eta[dst]:
                    min_eta[dst] = eta
            star_x[:, 18] = np.log1p(inbound) / 5.0
            star_x[:, 19] = np.log1p(min_eta) / 4.0

        # Col 29 (`ticks_to_enemy`) & Col 31 (`in_enemy_jump`): focus on HOSTILE stars
        # (falling back to deterrence borders only when 0 HOSTILE stars exist).
        focus_idx = np.flatnonzero(is_hostile_star)
        if len(focus_idx) == 0 and neutral_border_posture != "open":
            focus_idx = np.flatnonzero(is_protected_star)

        if len(focus_idx) > 0:
            t_to_e = np.minimum(static.travel[:, focus_idx].min(axis=1), max_t)
            hs_all = np.array(
                [int(static.hyperspace(levels, q)) for q in range(P)],
                dtype=np.int64,
            )
            e_r_ticks = (
                static.hyper_range[
                    np.clip(hs_all, 1, len(static.hyper_range)) - 1
                ]
                / max(static.speed, 1e-4)
            )
            in_e_jump = (
                static.travel[focus_idx]
                <= e_r_ticks[owner[focus_idx]][:, None]
            ).any(axis=0).astype(np.float32)
        else:
            t_to_e = np.full(S, max_t, dtype=np.float32)
            in_e_jump = np.zeros(S, dtype=np.float32)

        star_x[:, 29] = np.log1p(t_to_e) / 4.0
        star_x[:, 31] = in_e_jump

    if research_allowed_mask is not None and len(research_allowed_mask) == len(
        np_in["research_mask"]
    ):
        merged = np_in["research_mask"] & np.asarray(
            research_allowed_mask, dtype=bool
        )
        if merged[1:].any():
            np_in["research_mask"] = merged

    if banned_star_specialists:
        for sid in banned_star_specialists:
            s_i = int(sid)
            if 1 <= s_i <= MAX_SPECIALIST_ID:
                np_in["spec_mask"][s_i] = False

    return np_in


def sanitize_decision(
    decision: dict,
    frame: dict,
    seat: int,
    diplomatic_tiers: list[str] | None,
    contested_unowned_stars: list[int] | None,
) -> dict:
    S = len(frame["owner"])
    owner = np.asarray(frame["owner"], dtype=np.int64)
    decision["court"] = False
    decision["betray"] = False

    if diplomatic_tiers and len(diplomatic_tiers) > 0:
        desire = decision.get("attackDesire")
        if desire is None:
            desire = [1.0] * S
        else:
            desire = list(desire)
        contested_set = {int(x) for x in (contested_unowned_stars or [])}
        for i in range(S):
            q = int(owner[i])
            if i in contested_set and q < 0:
                desire[i] = 0.0
            elif q >= 0 and q != seat and diplomatic_tiers[q] != "HOSTILE":
                desire[i] = 0.0
        decision["attackDesire"] = desire

    return decision


class LiveRpcServer:
    def __init__(self):
        self.static: bd.Static | None = None
        self.seat: int = 0
        self.model: AdmiralNet | None = None
        self.checkpoint_loaded: str | None = "__unloaded__"
        self.tracker: BeliefTracker | None = None
        self.device = torch.device("cpu")

    def cmd_init(self, args: dict) -> dict:
        header = args["header"]
        self.seat = int(args["seat"])
        ckpt = args.get("checkpoint") or None
        self.static = bd.Static(header)
        n_techs = len(self.static.techs)

        if self.model is None or ckpt != self.checkpoint_loaded or self.model.n_techs != n_techs:
            self.model = _load_compatible_checkpoint(n_techs, ckpt).to(
                self.device
            )
            self.checkpoint_loaded = ckpt

        self.tracker = BeliefTracker(self.static, self.seat)
        if args.get("trackerState"):
            restore_tracker(self.tracker, args["trackerState"])

        return {
            "stars": self.static.S,
            "players": self.static.P,
            "techs": self.static.techs,
            "checkpoint": ckpt,
        }

    def cmd_restore_tracker(self, args: dict) -> dict:
        if self.tracker is None:
            raise RuntimeError("Call init before restore_tracker")
        restore_tracker(self.tracker, args.get("state"))
        return {"lastTick": self.tracker._last_tick, "historyLen": len(self.tracker.history)}

    def cmd_ingest_frames(self, args: dict) -> dict:
        if self.tracker is None:
            raise RuntimeError("Call init before ingest_frames")
        frames = args.get("frames") or []
        for fr in frames:
            self.tracker.update(fr)
        return {
            "ingested": len(frames),
            "lastTick": self.tracker._last_tick,
            "historyLen": len(self.tracker.history),
        }

    @torch.no_grad()
    def cmd_decide(self, args: dict) -> dict:
        if self.static is None or self.model is None or self.tracker is None:
            raise RuntimeError("Call init before decide")

        obs = args["obs"]
        frame = obs["frame"]
        own_facts = obs.get("ownFacts")
        ticks_to_end = obs.get("ticksToEnd")
        tiers = args.get("diplomaticTiers")
        neutral_border = args.get("neutralBorderPosture", "defend")
        res_mask = args.get("researchAllowedMask")
        banned_specs = args.get("bannedStarSpecialists")
        contested_stars = args.get("contestedUnownedStars")

        np_in = build_admiral_inputs(
            self.static,
            frame,
            self.seat,
            own_facts=own_facts,
            tracker=self.tracker,
            ticks_to_end=ticks_to_end,
        )
        apply_diplomatic_adaptation(
            np_in,
            self.static,
            frame,
            self.seat,
            diplomatic_tiers=tiers,
            neutral_border_posture=neutral_border,
            research_allowed_mask=res_mask,
            banned_star_specialists=banned_specs,
            contested_unowned_stars=contested_stars,
        )

        batch = _to_torch_batch([np_in], self.device)
        res = self.model.act_or_evaluate(batch, sample=False, temperature=1.0)
        act_np = {
            k: v[0].detach().cpu().numpy() for k, v in res["actions"].items()
        }
        belief_np = (
            res["policy_out"]["belief_logits"][0].detach().cpu().numpy()
        )
        res_logits = (
            res["policy_out"]["res_logits"][0].detach().cpu().numpy().tolist()
        )
        fog_value = float(res["policy_out"]["fog_value"][0].item())

        decision = decode_admiral_decision(
            self.static,
            frame,
            self.seat,
            act_np,
            belief_logits=belief_np,
            tracker=self.tracker,
            send_beliefs=True,
        )
        decision = sanitize_decision(
            decision, frame, self.seat, tiers, contested_stars
        )

        return {
            "decision": decision,
            "resLogits": [round(float(x), 5) for x in res_logits],
            "fogValue": round(fog_value, 5),
            "trackerState": serialize_tracker(self.tracker),
        }

    @torch.no_grad()
    def cmd_propose_search(self, args: dict) -> dict:
        """Generates `nPlans` diplomatically-masked candidate plans + `nWorlds`
        conservation-exact `BeliefWorld`s for 1-cycle lookahead at cycle start."""
        if self.static is None or self.model is None or self.tracker is None:
            raise RuntimeError("Call init before propose_search")

        obs = args["obs"]
        frame = obs["frame"]
        own_facts = obs.get("ownFacts")
        ticks_to_end = obs.get("ticksToEnd")
        tiers = args.get("diplomaticTiers")
        neutral_border = args.get("neutralBorderPosture", "defend")
        res_mask = args.get("researchAllowedMask")
        banned_specs = args.get("bannedStarSpecialists")
        contested_stars = args.get("contestedUnownedStars")
        n_plans = max(2, int(args.get("nPlans", 4)))
        n_worlds = max(1, int(args.get("nWorlds", 2)))
        seed = int(args.get("seed", 1))
        temp = float(args.get("temperature", 1.15))

        np_in = build_admiral_inputs(
            self.static,
            frame,
            self.seat,
            own_facts=own_facts,
            tracker=self.tracker,
            ticks_to_end=ticks_to_end,
        )
        apply_diplomatic_adaptation(
            np_in,
            self.static,
            frame,
            self.seat,
            diplomatic_tiers=tiers,
            neutral_border_posture=neutral_border,
            research_allowed_mask=res_mask,
            banned_star_specialists=banned_specs,
            contested_unowned_stars=contested_stars,
        )

        batch_1 = _to_torch_batch([np_in], self.device)
        greedy_res = self.model.act_or_evaluate(batch_1, sample=False)
        belief_np = (
            greedy_res["policy_out"]["belief_logits"][0].detach().cpu().numpy()
        )
        res_logits = (
            greedy_res["policy_out"]["res_logits"][0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        candidates_act = [
            {
                k: v[0].detach().cpu().numpy()
                for k, v in greedy_res["actions"].items()
            }
        ]
        n_extra = n_plans - 1
        if n_extra > 0:
            batch_m = _to_torch_batch([np_in] * n_extra, self.device)
            samp_res = self.model.act_or_evaluate(
                batch_m, sample=True, temperature=temp
            )
            for m in range(n_extra):
                candidates_act.append(
                    {
                        k: v[m].detach().cpu().numpy()
                        for k, v in samp_res["actions"].items()
                    }
                )

        plans = [
            sanitize_decision(
                decode_admiral_decision(
                    self.static,
                    frame,
                    self.seat,
                    act_d,
                    belief_logits=belief_np,
                    tracker=self.tracker,
                    send_beliefs=True,
                ),
                frame,
                self.seat,
                tiers,
                contested_stars,
            )
            for act_d in candidates_act
        ]

        tick = int(frame["tick"])
        rng = np.random.default_rng(seed * 1009 + tick * 37 + self.seat)
        worlds = self.tracker.sample_belief_worlds(
            frame,
            k_worlds=n_worlds,
            rng=rng,
            learned_ship_logits=belief_np[:, 0],
            learned_ind_logits=belief_np[:, 2],
        )

        return {
            "plans": plans,
            "worlds": worlds,
            "resLogits": [round(float(x), 5) for x in res_logits],
            "trackerState": serialize_tracker(self.tracker),
        }

    @torch.no_grad()
    def cmd_score_leaves(self, args: dict) -> dict:
        if self.static is None or self.model is None:
            raise RuntimeError("Call init before score_leaves")

        leaves = args["leaves"]
        tiers = args.get("diplomaticTiers")
        neutral_border = args.get("neutralBorderPosture", "defend")
        leaf_inputs = []
        leaf_shares = []

        for leaf in leaves:
            lf_frame = leaf["frame"]
            lf_own = leaf.get("ownFacts")
            rem_end = leaf.get("ticksToEnd")
            inp = build_admiral_inputs(
                self.static,
                lf_frame,
                self.seat,
                own_facts=lf_own,
                tracker=None,
                ticks_to_end=rem_end,
            )
            apply_diplomatic_adaptation(
                inp,
                self.static,
                lf_frame,
                self.seat,
                diplomatic_tiers=tiers,
                neutral_border_posture=neutral_border,
            )
            leaf_inputs.append(inp)

            players = leaf["players"]
            tot_stars = max(1, sum(p["stars"] for p in players))
            tot_ships = max(1, sum(p["ships"] for p in players))
            me = players[self.seat]
            share = (
                0.6 * (me["stars"] / tot_stars)
                + 0.4 * (me["ships"] / tot_ships)
                - (1.0 if me["defeated"] else 0.0)
            )
            leaf_shares.append(share)

        batch = _to_torch_batch(leaf_inputs, self.device)
        out = self.model.forward_policy(batch)
        fog_vals = out["fog_value"].detach().cpu().numpy()
        combined = fog_vals + 0.25 * np.asarray(leaf_shares, dtype=np.float32)
        return {"scores": [float(x) for x in combined.tolist()]}


def main():
    server = LiveRpcServer()
    dispatch = {
        "init": server.cmd_init,
        "restore_tracker": server.cmd_restore_tracker,
        "ingest_frames": server.cmd_ingest_frames,
        "decide": server.cmd_decide,
        "propose_search": server.cmd_propose_search,
        "score_leaves": server.cmd_score_leaves,
    }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            cmd = req.get("cmd")
            if cmd not in dispatch:
                raise ValueError(f"Unknown RPC command '{cmd}'")
            result = dispatch[cmd](req.get("args") or {})
            sys.stdout.write(
                json.dumps({"id": req_id, "ok": True, "result": result}) + "\n"
            )
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"id": req_id, "ok": False, "error": str(exc)})
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
