"""Persistent-worker rollout collector and cross-play evaluator (`Admiral` <-> `Commander`).

Supports:
1. Persistent `AdmiralEnv` Node subprocess per worker (zero Node startup cost
   between games) + cached checkpoint loading.
2. Native cross-play in the same Solaris galaxy: opponent seats can be frozen
   `Admiral` checkpoints, frozen `Commander` checkpoints from `data/league1/`,
   persona-tuned AIs (`personas.py`), or the standard built-in AI.
3. Optional `.sbd` belief-recording output (`SBD1` format shared with
   `server/sim/python/belief_data.py`).
4. Optional `AdmiralSearchPlanner` 1-cycle fog-legal lookahead during evaluation.
"""

import argparse
import atexit
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

_UPSTREAM_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python")
)
if _UPSTREAM_PY not in sys.path:
    sys.path.insert(0, _UPSTREAM_PY)

import belief_data as bd  # noqa: E402
import commander as cm  # noqa: E402
import commander_rollout as cr  # noqa: E402
import personas as pa  # noqa: E402
import presets  # noqa: E402

from admiral_agent import AdmiralAgent, AdmiralSearchPlanner  # noqa: E402
from admiral_env import AdmiralEnv  # noqa: E402
from admiral_net import (  # noqa: E402
    HEAD_GROUPS,
    AdmiralNet,
    build_admiral_inputs,
)
from belief_tracker import BeliefTracker  # noqa: E402

STAR_SHARE_BONUS = 0.02
SHIP_SHARE_BONUS = presets.PRESET["ship_share_bonus"]
PERSONA_POOLS = {
    "train": pa.TRAIN,
    "held_out": pa.HELD_OUT,
    "all": pa.ALL,
}

_WORKER_ENV: AdmiralEnv | None = None
_MODEL_CACHE: dict[tuple[str, int, int], tuple[str, any]] = {}


def _get_worker_env() -> AdmiralEnv:
    global _WORKER_ENV
    if _WORKER_ENV is None:
        _WORKER_ENV = AdmiralEnv()
        atexit.register(_close_worker_env)
    return _WORKER_ENV


def _close_worker_env():
    global _WORKER_ENV
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()
        _WORKER_ENV = None


def placement_reward(place: int, players: int) -> float:
    if players <= 1:
        return 0.0
    return 1.0 - 2.0 * float(place - 1) / float(players - 1)


def placements(standings: dict) -> dict[int, int]:
    key = lambda p: (
        standings["winner"] != p["seat"],
        p["defeated"],
        -p["stars"],
        -p["ships"],
    )
    order = sorted(standings["players"], key=key)
    return {p["seat"]: i + 1 for i, p in enumerate(order)}


def load_admiral_model(
    n_techs: int,
    checkpoint: str | Path | None = None,
    d: int = 128,
    n_layers: int = 4,
) -> AdmiralNet:
    if checkpoint is None:
        model = AdmiralNet(n_techs, d=d, n_layers=n_layers)
        model._admiral_iter = 0
        model.eval()
        return model

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved.get("config", {})
    model = AdmiralNet(
        n_techs=cfg.get("n_techs", n_techs),
        d=cfg.get("d", d),
        n_layers=cfg.get("n_layers", n_layers),
        n_heads=cfg.get("n_heads", 4),
    )
    state = saved["model"] if "model" in saved else saved
    model.load_state_dict(state, strict=True)
    model._admiral_iter = int(saved.get("iteration", 0)) if isinstance(saved, dict) else 0
    model.eval()
    return model


def load_any_checkpoint_cached(path: str | Path, n_techs: int) -> tuple[str, any]:
    """Auto-detect whether `path` is an `Admiral` checkpoint or a legacy
    `Commander` checkpoint, caching the loaded `nn.Module` per worker."""
    p_str = os.path.abspath(str(path))
    mtime = os.stat(p_str).st_mtime_ns
    cache_key = (p_str, mtime, int(n_techs))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Bound worker cache size so long league runs never leak RAM
    if len(_MODEL_CACHE) >= 24:
        _MODEL_CACHE.clear()

    saved = torch.load(p_str, map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and str(saved.get("arch", "")).startswith("admiral"):
        cfg = saved.get("config", {})
        model = AdmiralNet(
            n_techs=cfg.get("n_techs", n_techs),
            d=cfg.get("d", 128),
            n_layers=cfg.get("n_layers", 4),
            n_heads=cfg.get("n_heads", 4),
        )
        model.load_state_dict(saved["model"], strict=True)
        model._admiral_iter = int(saved.get("iteration", 0))
        model.eval()
        entry = ("admiral", model)
    else:
        model = cr.build_model(n_techs, None, p_str)
        model.eval()
        entry = ("commander", model)

    _MODEL_CACHE[cache_key] = entry
    return entry


def _normalize_opponent_spec(spec) -> tuple[str | None, str]:
    """Accept `'path/to/ckpt.pt'` or `{'kind': 'admiral'|'commander', 'path': ...}`."""
    if isinstance(spec, dict):
        return spec.get("kind"), str(spec["path"])
    return None, str(spec)


def _observe_commander_seat(env: AdmiralEnv, agent: cm.CommanderAgent, seat: int) -> dict:
    if getattr(env, "cycle_only", False):
        return env.observe(seat, frame=True, truth_frame=False)
    obs = env.observe(seat)
    if agent.wants_frame(obs, seat):
        obs = env.observe(seat, frame=True, truth_frame=False)
    return obs


def play_admiral_game(
    seed: int,
    seats: int = presets.SEATS,
    checkpoint: str | Path | None = None,
    max_ticks: int = 3000,
    learner_seats: list[int] | None = None,
    opponents: dict[int, any] | None = None,
    record: str | None = None,
    ai_configs: dict[int, dict] | None = None,
    cycle_only: bool = True,
    env: AdmiralEnv | None = None,
) -> dict[str, np.ndarray]:
    """Play one training rollout game with the learner `AdmiralNet` in
    `learner_seats`, against any combination of frozen `Admiral` and frozen
    `Commander` checkpoints in `opponents` plus built-in AI in remaining seats."""
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    opponents = {int(k): v for k, v in (opponents or {}).items()}
    if learner_seats is None:
        avail = [s for s in range(presets.PLAYERS) if s not in opponents]
        learner_seats = sorted(
            int(s) for s in rng.choice(avail, min(seats, len(avail)), replace=False)
        )
    else:
        learner_seats = sorted(int(s) for s in learner_seats)

    own_env = env is None
    if env is None:
        env = _get_worker_env()

    # Probe header / inspect opponent checkpoint kinds before reset so we pass
    # exact `admiral_seats` vs `commander_seats`.
    opp_paths: dict[int, str] = {}
    opp_kinds: dict[int, str] = {}
    for s, spec in opponents.items():
        hint_kind, p_str = _normalize_opponent_spec(spec)
        opp_paths[s] = p_str
        if hint_kind in ("admiral", "commander"):
            opp_kinds[s] = hint_kind
        else:
            hdr_peek = torch.load(p_str, map_location="cpu", weights_only=False)
            opp_kinds[s] = (
                "admiral"
                if isinstance(hdr_peek, dict)
                and str(hdr_peek.get("arch", "")).startswith("admiral")
                else "commander"
            )

    admiral_all = sorted(
        learner_seats + [s for s, k in opp_kinds.items() if k == "admiral"]
    )
    commander_all = sorted(s for s, k in opp_kinds.items() if k == "commander")
    all_agent_seats = sorted(admiral_all + commander_all)

    try:
        result = env.reset(
            admiral_seats=admiral_all,
            commander_seats=commander_all,
            settings=presets.NAME,
            seed=int(seed),
            max_ticks=int(max_ticks),
            fog=True,
            record=record,
            record_info={"generator": "admiral", "seed": int(seed)},
            ai_configs=presets.builtin_configs(all_agent_seats, ai_configs),
            cycle_only=bool(cycle_only),
        )
        static = bd.Static(env.header())
        n_techs = len(static.techs)

        if checkpoint:
            _, learner_model = load_any_checkpoint_cached(checkpoint, n_techs)
        else:
            learner_model = load_admiral_model(n_techs)
        learner = AdmiralAgent(learner_model, static, sample=True)

        opp_agents: dict[int, tuple[str, any]] = {}
        by_path_agent: dict[str, tuple[str, any]] = {}
        for s, p_str in opp_paths.items():
            if p_str not in by_path_agent:
                kind, m = load_any_checkpoint_cached(p_str, n_techs)
                ag = (
                    AdmiralAgent(m, static, sample=False)
                    if kind == "admiral"
                    else cm.CommanderAgent(m, static, sample=False)
                )
                by_path_agent[p_str] = (kind, ag)
            opp_agents[s] = by_path_agent[p_str]

        learner_set = set(learner_seats)
        truth_trackers = {s: BeliefTracker(static, s) for s in learner_seats}
        steps = []
        last_cycle: dict[int, int] = {}

        while not result["done"]:
            for seat in result["seats"]:
                seat = int(seat)
                if seat in opp_agents:
                    kind, opp_ag = opp_agents[seat]
                    if kind == "admiral":
                        obs = env.observe_admiral(seat, truth_frame=False)
                        dec, _ = opp_ag.decide(obs, seat, record_step=False)
                        env.act_admiral(seat, dec)
                    else:
                        obs = _observe_commander_seat(env, opp_ag, seat)
                        dec, _ = opp_ag.decide(obs, seat)
                        env.act(seat, dec)
                    continue

                if seat not in learner_set:
                    continue

                obs = env.observe_admiral(seat, truth_frame=True)
                dec, rec = learner.decide(obs, seat, record_step=True)
                env.act_admiral(seat, dec)

                truth_in = build_admiral_inputs(
                    static,
                    obs["truthFrame"],
                    seat,
                    own_facts=obs.get("ownFacts"),
                    tracker=truth_trackers[seat],
                    ticks_to_end=obs.get("ticksToEnd"),
                    truth_facts=obs.get("truthFacts"),
                )

                bonus = 0.0
                if obs["cycle"] != last_cycle.get(seat):
                    last_cycle[seat] = obs["cycle"]
                    own_cnt = sum(1 for o in obs["frame"]["owner"] if o == seat)
                    ships_all = [row[1] for row in obs["frame"]["stats"]]
                    bonus = (
                        STAR_SHARE_BONUS * own_cnt / max(static.S, 1)
                        + SHIP_SHARE_BONUS
                        * ships_all[seat]
                        / max(sum(ships_all), 1)
                    )

                aux_belief = {
                    "owner": np.asarray(obs["frame"]["owner"], dtype=np.int16),
                    "vis": np.asarray(obs["frame"]["visibility"], dtype=np.uint8),
                    "true_ships": np.asarray(
                        obs["truthFrame"]["mem_ships"], dtype=np.float32
                    ),
                    "true_infra": np.asarray(
                        obs["truthFrame"]["mem_infra"], dtype=np.float32
                    ),
                }
                steps.append((seat, int(obs["tick"]), rec, truth_in, aux_belief, bonus))

            result = env.step()

        standings = env.standings()
    finally:
        if own_env and env is not _WORKER_ENV:
            env.close()

    place = placements(standings)
    policy_iter = int(getattr(learner_model, "_admiral_iter", 0))
    return pack_admiral_rollout(
        steps, place, learner_seats, int(seed), standings, policy_iter=policy_iter
    )


def pack_admiral_rollout(
    steps: list,
    place: dict[int, int],
    learner_seats: list[int],
    seed: int,
    standings: dict,
    policy_iter: int = 0,
) -> dict[str, np.ndarray]:
    out = {
        "seat": [],
        "tick": [],
        "logp": [],
        "bonus": [],
    }
    for g in HEAD_GROUPS:
        out[f"logp_{g}"] = []

    inputs: dict[str, list[np.ndarray]] = {}
    acts: dict[str, list[np.ndarray]] = {}
    parts: dict[str, list[np.ndarray]] = {}
    truths: dict[str, list[np.ndarray]] = {}
    edge_table: list[np.ndarray] = []
    step_edge_idx: list[int] = []
    aux: dict[str, list[np.ndarray]] = {
        "aux_owner": [],
        "aux_vis": [],
        "aux_true_ships": [],
        "aux_true_infra": [],
    }

    for seat, tick, rec, truth_in, aux_b, bonus in steps:
        out["seat"].append(seat)
        out["tick"].append(tick)
        out["logp"].append(rec["logp"])
        out["bonus"].append(bonus)
        for g in HEAD_GROUPS:
            out[f"logp_{g}"].append(rec["logp_groups"][g])

        # Deduplicate static/range channels 0..2 of edge_x per game (~225x smaller .npz!)
        e3 = rec["inputs"]["edge_x"][:, :, :3].astype(np.float16)
        found_idx = -1
        for tbl_i, prev_e3 in enumerate(edge_table):
            if np.array_equal(e3[:, :, 2], prev_e3[:, :, 2]):
                found_idx = tbl_i
                break
        if found_idx < 0:
            found_idx = len(edge_table)
            edge_table.append(e3)
        step_edge_idx.append(found_idx)

        for k, v in rec["inputs"].items():
            if k == "edge_x":
                continue
            inputs.setdefault(k, []).append(v)
        for k, v in rec["act"].items():
            acts.setdefault(k, []).append(np.asarray(v))
        for k, v in rec.get("logp_parts", {}).items():
            parts.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        for k in ("star_x", "player_x", "owner_rel", "time_x"):
            truths.setdefault(k, []).append(truth_in[k])
        aux["aux_owner"].append(aux_b["owner"])
        aux["aux_vis"].append(aux_b["vis"])
        aux["aux_true_ships"].append(aux_b["true_ships"])
        aux["aux_true_infra"].append(aux_b["true_infra"])

    arrays = {k: np.asarray(v) for k, v in out.items()}
    if steps:
        arrays.update({"in_" + k: np.stack(v, axis=0) for k, v in inputs.items()})
        arrays.update({"act_" + k: np.stack(v, axis=0) for k, v in acts.items()})
        arrays.update({"lp_" + k: np.stack(v, axis=0) for k, v in parts.items()})
        arrays.update({"truth_" + k: np.stack(v, axis=0) for k, v in truths.items()})
        arrays.update({k: np.stack(v, axis=0) for k, v in aux.items()})
        arrays["edge_static"] = np.stack(edge_table, axis=0)  # [K, S, S, 3] float16
        arrays["step_edge_idx"] = np.asarray(step_edge_idx, dtype=np.int16)

    n_players = len(standings["players"])
    arrays["agents"] = np.asarray(learner_seats, dtype=np.int32)
    arrays["place"] = np.asarray([place[s] for s in learner_seats], dtype=np.int32)
    arrays["final_reward"] = np.asarray(
        [placement_reward(place[s], n_players) for s in learner_seats],
        dtype=np.float32,
    )
    arrays["hired"] = np.asarray(
        [standings["players"][s].get("hired", 0) for s in learner_seats],
        dtype=np.int32,
    )
    arrays["hires_refused"] = np.asarray(
        [standings["players"][s].get("hiresRefused", 0) for s in learner_seats],
        dtype=np.int32,
    )
    arrays["place_all"] = np.asarray(
        [place[p["seat"]] for p in standings["players"]], dtype=np.int32
    )
    arrays["n_players"] = np.asarray(n_players, dtype=np.int32)
    arrays["seed"] = np.asarray(seed, dtype=np.int32)
    arrays["ticks"] = np.asarray(standings["tick"], dtype=np.int32)
    arrays["policy_iter"] = np.asarray(policy_iter, dtype=np.int32)
    return arrays


def play_eval_game(
    seed: int,
    seats: int = presets.SEATS,
    checkpoint: str | Path | None = "untrained",
    max_ticks: int = 3000,
    personas: dict | None = None,
    opponent: str | Path | None = None,
    use_search: bool = False,
    search_plans: int = 4,
    search_worlds: int = 2,
    cycle_only: bool = True,
    env: AdmiralEnv | None = None,
) -> dict[int, int]:
    """Evaluate `Admiral` on `seed`'s evaluation seats against built-in AI,
    personas, or a head-to-head `opponent` checkpoint (`Commander` OR `Admiral`!).

    - `checkpoint=None` runs pure built-in AI in all seats (full-view bar).
    - `checkpoint="untrained"` runs an untrained zero-init `AdmiralNet` +
      `TacticalExecutor` under fog.
    - `opponent` plays `seats` rival seats; rival placements are returned under
      negative keys `-1 - seat` (matching `commander_rollout.play_eval`).
    """
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    agents = cr.eval_seats(int(seed), seats)
    rng = np.random.default_rng(int(seed) + 7_919)
    others = [s for s in rng.permutation(presets.PLAYERS).tolist() if s not in agents]
    rivals = sorted(others[:seats]) if opponent else []

    ai_configs = {}
    if personas:
        for s, (_, cfg) in pa.assign(
            rng, [s for s in others if s not in rivals], personas
        ).items():
            ai_configs[s] = cfg

    own_env = env is None
    if env is None:
        env = _get_worker_env()

    opp_kind = None
    if opponent:
        hdr = torch.load(opponent, map_location="cpu", weights_only=False)
        opp_kind = (
            "admiral"
            if isinstance(hdr, dict) and str(hdr.get("arch", "")).startswith("admiral")
            else "commander"
        )

    admiral_seats = list(agents) if checkpoint is not None else []
    commander_seats = []
    if opponent:
        if opp_kind == "admiral":
            admiral_seats = sorted(admiral_seats + rivals)
        else:
            commander_seats = sorted(rivals)
    all_played = sorted(admiral_seats + commander_seats)

    try:
        result = env.reset(
            admiral_seats=admiral_seats,
            commander_seats=commander_seats,
            settings=presets.NAME,
            seed=int(seed),
            max_ticks=int(max_ticks),
            fog=True,
            ai_configs=presets.builtin_configs(all_played, ai_configs),
            cycle_only=bool(cycle_only),
        )
        static = bd.Static(env.header())
        n_techs = len(static.techs)

        main_agent = None
        planner = None
        if checkpoint is not None:
            if checkpoint == "untrained":
                m = load_admiral_model(n_techs, None)
            else:
                _, m = load_any_checkpoint_cached(checkpoint, n_techs)
            main_agent = AdmiralAgent(m, static, sample=False)
            if use_search:
                planner = AdmiralSearchPlanner(
                    main_agent, n_plans=search_plans, n_worlds=search_worlds
                )

        rival_agent = None
        if opponent:
            _, rm = load_any_checkpoint_cached(opponent, n_techs)
            rival_agent = (
                AdmiralAgent(rm, static, sample=False)
                if opp_kind == "admiral"
                else cm.CommanderAgent(rm, static, sample=False)
            )

        agent_set = set(agents)
        rival_set = set(rivals)

        while not result["done"]:
            for seat in result["seats"]:
                seat = int(seat)
                if seat in agent_set and main_agent is not None:
                    obs = env.observe_admiral(seat, truth_frame=False)
                    if planner is not None:
                        dec = planner.decide(env, obs, seat, seed=int(seed))
                    else:
                        dec, _ = main_agent.decide(obs, seat, record_step=False)
                    env.act_admiral(seat, dec)
                elif seat in rival_set and rival_agent is not None:
                    if opp_kind == "admiral":
                        obs = env.observe_admiral(seat, truth_frame=False)
                        dec, _ = rival_agent.decide(obs, seat, record_step=False)
                        env.act_admiral(seat, dec)
                    else:
                        obs = _observe_commander_seat(env, rival_agent, seat)
                        dec, _ = rival_agent.decide(obs, seat)
                        env.act(seat, dec)
            result = env.step()

        place = placements(env.standings())
    finally:
        if own_env and env is not _WORKER_ENV:
            env.close()

    out = {s: place[s] for s in agents}
    out.update({-1 - s: place[s] for s in rivals})
    return out


def _rollout_worker(args):
    if len(args) == 10:
        (
            out_dir,
            seed,
            seats,
            checkpoint,
            max_ticks,
            learner_seats,
            opponents,
            record_path,
            ai_configs,
            cycle_only,
        ) = args
    else:
        (
            out_dir,
            seed,
            seats,
            checkpoint,
            max_ticks,
            opponents,
            record_path,
            ai_configs,
            cycle_only,
        ) = args
        learner_seats = None
    path = Path(out_dir) / f"r{seed}.npz"
    if path.exists():
        return seed, None
    arrays = play_admiral_game(
        seed=seed,
        seats=seats,
        checkpoint=checkpoint,
        max_ticks=max_ticks,
        learner_seats=learner_seats,
        opponents=opponents,
        record=record_path,
        ai_configs=ai_configs,
        cycle_only=cycle_only,
    )
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return seed, arrays["place"].tolist()


def _async_rollout_worker(args):
    """Continuous async rollout worker: loads `checkpoint` (typically
    `ckpt_latest.pt`, reloaded automatically whenever its mtime changes), plays
    one game, writes `<out_dir>/r<seed>.npz`, and returns metadata."""
    (
        out_dir,
        seed,
        seats,
        checkpoint,
        max_ticks,
        learner_seats,
        opponents,
        record_path,
        ai_configs,
        cycle_only,
    ) = args
    path = Path(out_dir) / f"r{seed}.npz"
    arrays = play_admiral_game(
        seed=seed,
        seats=seats,
        checkpoint=checkpoint,
        max_ticks=max_ticks,
        learner_seats=learner_seats,
        opponents=opponents,
        record=record_path,
        ai_configs=ai_configs,
        cycle_only=cycle_only,
    )
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return (
        int(seed),
        str(path),
        int(arrays["policy_iter"]),
        arrays["place"].tolist(),
    )


def _eval_worker(args):
    (
        seed,
        seats,
        checkpoint,
        max_ticks,
        personas,
        opponent,
        use_search,
        search_plans,
        search_worlds,
        cycle_only,
    ) = args
    return seed, play_eval_game(
        seed=seed,
        seats=seats,
        checkpoint=checkpoint,
        max_ticks=max_ticks,
        personas=personas,
        opponent=opponent,
        use_search=use_search,
        search_plans=search_plans,
        search_worlds=search_worlds,
        cycle_only=cycle_only,
    )


def evaluate_suite(
    seeds: list[int],
    checkpoint: str | Path | None = "untrained",
    opponent: str | Path | None = None,
    personas_name: str | None = None,
    workers: int = 4,
    seats: int = presets.SEATS,
    max_ticks: int = 3000,
    use_search: bool = False,
    search_plans: int = 4,
    search_worlds: int = 2,
    cycle_only: bool = True,
) -> dict:
    persona_pool = PERSONA_POOLS[personas_name] if personas_name else None
    jobs = [
        (
            int(s),
            seats,
            checkpoint,
            max_ticks,
            persona_pool,
            opponent,
            use_search,
            search_plans,
            search_worlds,
            cycle_only,
        )
        for s in seeds
    ]
    adm_places = []
    opp_places = []
    if workers <= 1:
        results = [_eval_worker(j) for j in jobs]
    else:
        with Pool(min(workers, len(jobs))) as pool:
            results = list(pool.imap_unordered(_eval_worker, jobs))

    for _, res in sorted(results, key=lambda x: x[0]):
        for k, p in res.items():
            if k >= 0:
                adm_places.append(p)
            else:
                opp_places.append(p)

    summary = {
        "games": len(seeds),
        "admiral_seats_total": len(adm_places),
        "admiral_mean_place": float(np.mean(adm_places)),
        "admiral_win_rate": float(np.mean([p == 1 for p in adm_places])),
        "admiral_top3_rate": float(np.mean([p <= 3 for p in adm_places])),
    }
    if opp_places:
        summary.update(
            {
                "opponent_mean_place": float(np.mean(opp_places)),
                "opponent_win_rate": float(np.mean([p == 1 for p in opp_places])),
                "opponent_top3_rate": float(np.mean([p <= 3 for p in opp_places])),
                "place_advantage": float(
                    np.mean(opp_places) - np.mean(adm_places)
                ),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Admiral vs built-in AI, personas, or a Commander/Admiral checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="untrained",
        help="Path to Admiral .pt checkpoint, 'untrained', or 'builtin' (default: untrained).",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        help="Optional opponent .pt checkpoint (Commander e.g. data/league1/ema_*.pt OR Admiral).",
    )
    parser.add_argument("--personas", default=None, choices=list(PERSONA_POOLS.keys()))
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--first-seed", type=int, default=900_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--search", action="store_true", help="Enable 1-cycle lookahead planner.")
    parser.add_argument("--search-plans", type=int, default=4)
    parser.add_argument("--search-worlds", type=int, default=2)
    args = parser.parse_args()

    ckpt = None if args.checkpoint == "builtin" else args.checkpoint
    seeds = list(range(args.first_seed, args.first_seed + args.games))
    summary = evaluate_suite(
        seeds=seeds,
        checkpoint=ckpt,
        opponent=args.opponent,
        personas_name=args.personas,
        workers=args.workers,
        max_ticks=args.max_ticks,
        use_search=args.search,
        search_plans=args.search_plans,
        search_worlds=args.search_worlds,
    )
    print(summary)


if __name__ == "__main__":
    main()
