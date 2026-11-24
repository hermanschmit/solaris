# Live Diplomatic `Admiral` Daemon (`server/sim/admiral/live/`)

Live human-in-the-loop bridge for running **`Admiral`** (`TacticalExecutor` + `AdmiralNet` + `BeliefTracker`) alongside a human operator in **10-Player Standard FFA** (`standard.json`) Solaris games.

---

## Key Design Guarantees

1. **100% Decoupled from `AdmiralNet` Checkpoint Internals:**
   - All live-bridge logic lives under `server/sim/admiral/live/` and imports `AdmiralNet`, `build_admiral_inputs`, `decode_admiral_decision`, and `BeliefTracker` from `server/sim/admiral/python/` without modifying any training/rollout script.
   - New `ema_*.pt` / `ckpt_*.pt` checkpoints drop in directly via `--checkpoint <path>`.
   - Default `TacticalExecutor.executeTick` and `AdmiralEnv` behavior is bit-for-bit unchanged when `TacticalExecutionOptions` is omitted.
2. **Human Owns All Diplomacy, Chat & Outbound Trades:**
   - Zero code paths in `server/sim/admiral/live/` call `PUT /diplomacy/*`, `PUT /trade/*`, chat send endpoints, or `PUT /ready`.
3. **3-Tier Diplomatic Posture & 5 Safety Invariants:**
   - **`ALLIED`** (`actualStatus === "allies"`): zero attacks, warp-network transit allowed, incoming carriers ignored by defense panic, and **hard-blocked** from ever being overridden by `attack:` in Notes (`ALLY_IN_ATTACK_IGNORED`).
   - **`PROTECTED`** (`actualStatus === "neutral"` or pending unilateral `statusTo === "allies"` offer): zero attacks, **no warp transit through neutral stars**, C2C reverse-edge collision avoidance when `carrierToCarrierCombat === "enabled"`, unowned contested-star avoidance, deterrence border garrisons (`neutral_border: defend | open`), and local defense + `SURPRISE_ATTACK` alert if a neutral non-gift carrier attacks without war declaration.
   - **`HOSTILE`** (`actualStatus === "enemies"` or covert `attack: [Alias]` neutral override): full `Admiral` offensive & multi-origin Time-on-Target (ToT) execution.
   - **Peace Mid-Attack Cancellation:** When a `HOSTILE` player transitions to `PROTECTED` or `ALLIED`, orbiting delayed ToT carriers targeting their stars are cleared (`waypoints = []`), and mid-flight carriers (`orbiting == null`) have follow-on `waypoints[1..]` truncated while emitting an urgent `UNRECALLABLE_FLEET` alert.
4. **Shared-Seat GUI Coexistence:**
   - **Independent Session Login:** Logs in via `POST /api/auth/login` with its own session cookie so your browser tab is never logged out.
   - **Sequential Rate Limiter & `Idempotency-Key`:** Strict `concurrency = 1` queue (default `3 req/s`; use `2 req/s` for Two-Hybrid shared-IP Docker testing) with deterministic `Idempotency-Key` headers and automatic `400 ValidationError` stale-state abort + 1x re-plan.
   - **Human Carrier Pinning (`humanPinnedCarriers`):** Automatically protects `isGift === true`, `waypointsLooped === true`, `[H]`/`MANUAL`-prefixed, and GUI-created/modified carriers (plus a configurable idle grace window, default `2` ticks) so `Admiral` never strips ships off a carrier you are staging or gifting.
   - **Protected Cash Reserve & `ignoreBulkUpgrade`:** Shields `cashReserveCredits + scheduledActionCredits` from Tick 1 & Tick 19 infrastructure upgrades and respects per-star `ignoreBulkUpgrade` toggles.
   - **Instant Research Coordination:** Subscribes to Socket.IO `playerTechnologyReceived` to switch `researchingNow` within seconds when an ally trades you a tech, avoids researching techs an in-range formal ally already holds higher (`TRADE_OPPORTUNITY`), and respects manual GUI research selections (`humanResearchPin`).
   - **SQLite / JSON Persistence & Time-Machine Backfill:** Persists `SeatMemory`, `BeliefTracker`, `aiCommittedCarriers`, `humanPinnedCarriers`, and `dispatchedIdempotencyKeys` across restarts, and warms up fog/belief history via `GET /api/game/:gameId/galaxy?tick=T` when attaching mid-game.

---

## Operating Modes

| Mode | Behavior |
| :--- | :--- |
| `paused` | Updates & persists `SeatMemory` only; plans and dispatches **0** actions. |
| `advisor` | Runs full shadow hydration, `BeliefTracker`, `AdmiralNet`, and `CommandRecorder` in memory; prints/logs proposed actions and alerts with **0** live `PUT` writes. |
| `approval` | Prompts interactively in the terminal (`y/N`) before dispatching each tick's batch. |
| `autonomous` | Dispatches coalesced tactical & research actions automatically within each tick window. |

---

## Optional Private In-Game Notes Block

Everything works through standard Solaris GUI controls (Diplomacy, Trade, Research, Star Ignore Bulk Upgrade, Carrier Gift/Loop). For the 5 settings without native Solaris buttons, you may optionally place an `# === AI ===` block anywhere inside your private **In-Game Notes** (all text outside the markers is untouched):

```yaml
# === AI ===
mode: autonomous        # paused | advisor | approval | autonomous
cash_reserve: 150       # credits Admiral must keep unspent for manual tech/credit trades
attack: [PlayerAliasX]  # covertly treat listed neutral player(s) as HOSTILE without declareEnemy
expect_tech: [weapons]  # promised by an ally in chat; skip researching locally
neutral_border: defend  # defend (deterrence garrison) | open (unfortified)
# === END AI ===
```

---

## CLI Usage

Run from `server/` (or repo root):

```bash
# 1. Read-only Advisor dry-run against a game:
SOLARIS_API=http://localhost:3000 \
SOLARIS_EMAIL=alpha@solaris.local \
SOLARIS_PASSWORD='your-password' \
  npx tsx server/sim/admiral/live/cli.ts \
    --game <gameId> \
    --mode advisor \
    --checkpoint data/admiral3/ema_490.pt \
    --state data/live/state_alpha.db \
    --once

# 2. Continuous Autonomous daemon with 150cr reserve & 1-cycle lookahead search:
SOLARIS_API=http://localhost:3000 \
SOLARIS_EMAIL=alpha@solaris.local \
SOLARIS_PASSWORD='your-password' \
  npx tsx server/sim/admiral/live/cli.ts \
    --game <gameId> \
    --mode autonomous \
    --checkpoint data/admiral3/ema_490.pt \
    --cash-reserve 150 \
    --rps 2 \
    --search \
    --state data/live/state_alpha.db
```

---

## Verification & Cloud Testing

```bash
# TypeScript typecheck across server workspace
npm run check --workspace=server

# Existing Admiral simulation & training spec suite (bit-for-bit regression check)
npx tsx node_modules/jasmine/bin/jasmine.js server/sim/admiral/admiral.spec.ts
python3 server/sim/admiral/python/test_admiral.py

# Live Diplomatic Admiral unit, shadow-sim, SQLite & Python RPC spec suite
npx tsx node_modules/jasmine/bin/jasmine.js server/sim/admiral/live/liveAdmiral.spec.ts

# Full Two-Hybrid E2E validation (Docker mongo:8.0 + real Solaris Express/Socket.IO + 2 Hybrid seats + 8 AI)
npx tsx server/sim/admiral/live/e2eTwoHybridValidation.ts
```
