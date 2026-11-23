# Admiral: Next-Gen Parallel AI & Co-Refinement Stack for Solaris

`server/sim/admiral/` is a self-contained, zero-touch parallel simulation, tactical execution, belief-tracking, and reinforcement-learning stack built alongside `server/sim/` (`Commander`). It modifies **zero** existing server or simulator files, allowing both `Commander` and `Admiral` to train concurrently, play against each other in the same galaxy, and share `.sbd` (`SBD1`) belief recordings.

See [DESIGN.md](./DESIGN.md) for the full architecture, mathematical formulation, design trade-offs, benchmarks, and literature references.

## Architecture Overview

| Component | File | Purpose |
| :--- | :--- | :--- |
| **O(1) / O(log N) Combat Sizing** | [fastDefense.ts](./fastDefense.ts) | Exact `fastFight` bulk-round division & binary-search `fastEstimateNeeded` matching `CombatService` bit-for-bit without 1-ship linear loops. |
| **Tactical Solver Executor** | [tacticalExecutor.ts](./tacticalExecutor.ts) | Hybrid synchronized Time-on-Target (`delayTicks <= 8`) + single-armada multi-hop invasions, visibility-aware fog attack margins, context-aware doomed-star triage & evacuation, specialist-weighted cycle infrastructure/Warp-Gate ROI upgrades, policy-driven `cycleEcoHold` carrier liquidity, and returning-ferry/`collectAll` logistics. |
| **Multi-Protocol Env & Lookahead** | [admiralEnv.ts](./admiralEnv.ts) | Subclasses `AgentEnv` to host `admiralSeats` and `commanderSeats` side-by-side under fog of war, expose calculator `ownFacts`, and run fog-legal `lookaheadAdmiral` across `BeliefWorld` determinizations. |
| **JSON-Lines RPC Server** | [server.ts](./server.ts) | Stdio RPC server supporting single & batched observations/actions for both `Admiral` and `Commander`. |
| **Conservation & Delta Belief Tracker** | [belief_tracker.py](./python/belief_tracker.py) | Enforces exact per-opponent scoreboard conservation (`PublicTotal - VisibleKnown`) for hidden ships & infrastructure, tracks rolling `Delta stats` combat attrition, and samples integer-exact `BeliefWorld`s. |
| **Heterogeneous Graph Transformer (`admiral_v3`)** | [admiral_net.py](./python/admiral_net.py) | `log1p`-scaled Star + Empire coupled Graph Transformer with travel-tick/hyperspace attention biases, continuous per-frontier-star candidate scoring (`frontier_head`), strategic postures (`balanced`, `expand`, `conquer`, `fortify`, `warp_surge`), 5-way Dirichlet cycle budget, feature-conditioned `research` (unshackled late-game `weapons`/`manufacturing`), conservation-residual `belief_head`, and 10-rank distributional critics (`truth` + `fog`). |
| **Supervised `.sbd` Pre-Trainer** | [admiral_pretrain.py](./python/admiral_pretrain.py) | Offline multi-task supervised pre-training of `backbone`, `critic_backbone`, `belief_head`, `fog_critic`, and `truth_critic` from `.sbd` (`SBD1`) recordings before PPO. |
| **Agent & Search Planner** | [admiral_agent.py](./python/admiral_agent.py) | `AdmiralAgent` + `AdmiralSearchPlanner` (1-cycle fog-legal belief-determinized lookahead). |
| **Cross-Play Rollouts & Eval** | [admiral_rollout.py](./python/admiral_rollout.py) | Persistent-Node worker pool supporting native cross-play against frozen `Commander` (`data/league1/ema_*.pt`) and `Admiral` checkpoints. |
| **Two-Timescale Per-Part PPO & League** | [admiral_train.py](./python/admiral_train.py) | Two-timescale trunk/head AdamW (`LR_TRUNK = 1.5e-5`, `LR_HEADS = 8e-5`), independent per-part ratio clipping (`PART_NAMES`) with `DUAL_CLIP = 3.0`, 4-channel conservation residual supervision, and 3-tier PFSP + `Commander` cross-play league. |

## Quick Commands (run from `server/`)

### 1. Run TypeScript & Python Tests
```bash
# TypeScript Jasmine specs
npx tsx ../node_modules/jasmine/bin/jasmine.js sim/admiral/admiral.spec.ts

# Python end-to-end verification (including .sbd pre-training -> --pretrain PPO init)
python3 sim/admiral/python/test_admiral.py
```

### 2. Stage 1 — Pre-Train `AdmiralNet` Backbone & Critics on `.sbd` Recordings
```bash
python3 sim/admiral/python/admiral_pretrain.py \
    --sbd-dirs data/admiral2/belief data/admiral1/belief data/league1/belief \
    --max-games 600 \
    --samples-per-game 24 \
    --epochs 10 \
    --workers 8 \
    --out data/admiral_belief.pt
```

### 3. Stage 2 — Train Admiral with `--pretrain` & Co-Refinement Cross-Play
```bash
python3 sim/admiral/python/admiral_train.py \
    --run-dir data/admiral3 \
    --pretrain data/admiral_belief.pt \
    --commander-dir data/league1 \
    --async-rollouts \
    --iters 500 \
    --games-per-iter 24 \
    --workers 12 \
    --record-every 4
```
- `--pretrain` loads the pre-trained `backbone`, `critic_backbone`, `belief_head`, `fog_critic`, and `truth_critic` on iter 0 while keeping all policy heads at their clean zero-initialized iter-0 priors, and updates the trunk at `LR_TRUNK = 1.5e-5` (`~5x` gentler than policy heads) so representations never collapse.
- Checkpoints (`ckpt_<i>.pt`, `ema_<i>.pt`), training metrics (`log.jsonl`), periodic benchmarks (`eval.jsonl`), and cross-play league ratings (`league.json`) are written to `--run-dir`.
- Re-running the same command automatically resumes from the highest numbered `ckpt_<i>.pt` in `--run-dir`.

### 4. Evaluate Admiral Head-to-Head vs Commander Checkpoints
```bash
# Single-pass EMA Admiral vs Commander EMA
python3 sim/admiral/python/admiral_rollout.py \
    --checkpoint data/admiral3/ema_450.pt \
    --opponent data/league1/ema_1117.pt \
    --games 24 --workers 8

# Enable 1-Cycle Fog-Legal Belief-Determinized Search Planner (--search)
python3 sim/admiral/python/admiral_rollout.py \
    --checkpoint data/admiral3/ema_490.pt \
    --opponent data/league1/ema_1117.pt \
    --search --search-plans 4 --search-worlds 2 \
    --games 24 --workers 8
```
