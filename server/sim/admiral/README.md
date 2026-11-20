# Admiral: Next-Gen Parallel AI & Co-Refinement Stack for Solaris

`server/sim/admiral/` is a self-contained, zero-touch parallel simulation, tactical execution, belief-tracking, and reinforcement-learning stack built alongside `server/sim/` (`Commander`). It modifies **zero** existing server or simulator files, allowing both `Commander` and `Admiral` to train concurrently, play against each other in the same galaxy, and share `.sbd` (`SBD1`) belief recordings.

## Architecture Overview

| Component | File | Purpose |
| :--- | :--- | :--- |
| **O(1) / O(log N) Combat Sizing** | [fastDefense.ts](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/fastDefense.ts) | Exact `fastFight` bulk-round division & binary-search `fastEstimateNeeded` matching `CombatService` bit-for-bit without 1-ship linear loops. |
| **Tactical Solver Executor** | [tacticalExecutor.ts](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/tacticalExecutor.ts) | Synchronized Time-on-Target (`delayTicks`) multi-star invasions, doomed-star triage & evacuation, targeted infrastructure/Warp-Gate ROI upgrades, multi-hop + Explorer claims, and shortest-tick Warp-Gate/Wormhole logistics. |
| **Multi-Protocol Env & Lookahead** | [admiralEnv.ts](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/admiralEnv.ts) | Subclasses `AgentEnv` to host `admiralSeats` and `commanderSeats` side-by-side under fog of war, expose calculator `ownFacts`, and run fog-legal `lookaheadAdmiral` across `BeliefWorld` determinizations. |
| **JSON-Lines RPC Server** | [server.ts](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/server.ts) | Stdio RPC server supporting single & batched observations/actions for both `Admiral` and `Commander`. |
| **Conservation & Delta Belief Tracker** | [belief_tracker.py](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/python/belief_tracker.py) | Enforces exact per-opponent scoreboard conservation (`PublicTotal - VisibleKnown`) for hidden ships & infrastructure, tracks rolling `Delta stats` combat attrition, and samples integer-exact `BeliefWorld`s. |
| **Heterogeneous Graph Transformer** | [admiral_net.py](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/python/admiral_net.py) | `log1p`-scaled Star + Empire coupled Graph Transformer with travel-tick/hyperspace attention biases, factored low-variance policy heads, auxiliary conservation-belief head, and 10-rank distributional placement critics (`truth` + `fog`). |
| **Agent & Search Planner** | [admiral_agent.py](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/python/admiral_agent.py) | `AdmiralAgent` + `AdmiralSearchPlanner` (1-cycle fog-legal belief-determinized lookahead). |
| **Cross-Play Rollouts & Eval** | [admiral_rollout.py](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/python/admiral_rollout.py) | Persistent-Node worker pool supporting native cross-play against frozen `Commander` (`data/league1/ema_*.pt`) and `Admiral` checkpoints. |
| **Factored PPO & Compact League** | [admiral_train.py](file://wsl.localhost/Ubuntu/home/herman/solaris/server/sim/admiral/python/admiral_train.py) | Per-head clipped PPO (`spatial`, `budget`, `knobs`, `macro`), distributional rank cross-entropy, auxiliary conservation-belief supervision, and 3-tier anti-dilution PFSP + `Commander` cross-play league. |

## Quick Commands (run from `server/`)

### 1. Run TypeScript & Python Tests
```bash
# TypeScript Jasmine specs
npx tsx ../node_modules/jasmine/bin/jasmine.js sim/admiral/admiral.spec.ts

# Python end-to-end verification
/home/herman/solaris-ml/bin/python3 sim/admiral/python/test_admiral.py
```

### 2. Evaluate Admiral Head-to-Head vs Commander Checkpoints
```bash
# Untrained Admiral (TacticalExecutor + Conservation Belief Prior) vs latest Commander EMA
/home/herman/solaris-ml/bin/python3 sim/admiral/python/admiral_rollout.py \
    --checkpoint untrained \
    --opponent data/league1/ema_1060.pt \
    --games 24 --workers 6

# Enable 1-Cycle Fog-Legal Belief-Determinized Search Planner (--search)
/home/herman/solaris-ml/bin/python3 sim/admiral/python/admiral_rollout.py \
    --checkpoint data/admiral1/ema_100.pt \
    --opponent data/league1/ema_1060.pt \
    --search --search-plans 4 --search-worlds 2 \
    --games 24 --workers 6
```

### 3. Train Admiral with Co-Refinement Cross-Play Against Commander
```bash
/home/herman/solaris-ml/bin/python3 sim/admiral/python/admiral_train.py \
    --run-dir data/admiral1 \
    --commander-dir data/league1 \
    --iters 300 \
    --games-per-iter 24 \
    --workers 8 \
    --record-every 4
```
- Checkpoints (`ckpt_<i>.pt`, `ema_<i>.pt`), training metrics (`log.jsonl`), periodic benchmarks (`eval.jsonl`), and cross-play league ratings (`league.json`) are written to `data/admiral1/`.
- Shared `SBD1` belief recordings are saved in `data/admiral1/belief/*.sbd` and can be fed directly into `sim/python/belief_data.py` to co-train `Commander`'s belief trunk as well.
