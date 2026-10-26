# Solaris simulator

Runs the server's own game logic in memory, with no database, web server or
sockets. It is the base for two tools:

- an **advisor** that takes your current view of a live game, fills in what
  you can't see, and plays candidate orders forward to compare them;
- an **arena** where agents play complete games against each other and the
  built-in AI.

Only the standard game mode has been exercised so far.

## How it works

`container.ts` builds the normal dependency container from
`services/index.ts`, then stubs the handful of services that would touch
Mongo during a tick (locking, history, events, notifications, statistics,
email). Upstream code is not modified. Two rules are changed for the sim:
a game made only of AI players does not end immediately, and ready-to-quit
is disabled on new games.

`engine.ts` drives `GameTickService._processTick` directly. Games are plain
objects rather than mongoose documents, which gives identical results and is
about five times faster. All of the server's randomness comes from
`Math.random`, so `engine.seed(n)` makes a run reproducible.

`snapshot.ts` turns the JSON from `GET /api/game/:gameId/galaxy` into a full
game. Hidden enemy infrastructure and ships are spread over their hidden
stars so that each player's published totals still add up; hidden carriers
are not invented. Different seeds give different plausible fills.

## Commands

From `server/` (build `common/` first with `npm run build -w common`):

```sh
# Time an all-AI standard game
npx tsx sim/cli.ts bench standard 100

# Save your view of a live game (see the file for auth options)
SOLARIS_COOKIE='...' npx tsx sim/fetchGalaxy.ts <gameId> snapshot.json

# Play an all-AI standard game to the end with seed 1
npx tsx sim/cli.ts play standard 4000 1

# Same, and record every tick to replay.json for the viewer
npx tsx sim/cli.ts play standard 4000 1 replay.json

# Fill in the fog and run it forward 20 ticks with seed 1
npx tsx sim/cli.ts step snapshot.json 20 1

# Tests
npx tsx ../node_modules/jasmine/bin/jasmine --filter="sim engine"
```

Set `LOGGING_LEVEL=fatal` to hide the built-in AI's error logs.

## Speed

Standard game, 10 players, 200 stars, on one core:

| Setup | Time per tick |
| --- | --- |
| Engine only (no AI players) | about 5 ms |
| All 10 players on the built-in advanced AI | about 90 to 135 ms |

Ticks get slower as carriers pile up. In a full all-AI game (seed 1,
measured with four games sharing four cores):

| Point in game | Carriers | With built-in AI | Engine only |
| --- | --- | --- | --- |
| Tick 100 | 194 | about 180 ms | about 27 ms |
| Tick 300 | 456 | about 420 ms | about 49 ms |

All-AI standard games end by conquest after roughly 400 to 600 ticks (20 to
30 cycles) and take two to four minutes on one core. The built-in AI is
about 85% of that time, so advisor rollouts and the arena will want a
cheaper opponent model.

## Watching a game

`play` with a file name records a replay (`replay.ts`): star owners and
ships, carrier positions and destinations, and each player's totals at every
tick. A full standard game is about 2 to 3 MB. Open `sim/viewer.html`
straight from disk in a browser and pick or drop the file. It shows the map
(stars sized by garrison, carriers in flight with their routes), the
standings at the current tick, and a chart of stars held over the game.
Space plays and pauses, the arrow keys step a tick (a cycle with shift), the
mouse wheel zooms and dragging pans.

## Tuning the built-in AI

The advanced AI's constants live in `AiConfig` (`services/types/Ai.ts`),
with `DEFAULT_AI_CONFIG` holding the values it has always used. The server
uses the defaults for everyone. In the sim, `engine.setAiConfig(game,
playerId, overrides)` gives one player its own tuning, for example a more
aggressive opponent:

```ts
engine.setAiConfig(game, player._id, {
    invasionAttackFactor: 1.2,
    orderPriorities: { defend: 3, invade: 4, claim: 2 },
});
```

## Driving seats with an agent

`agentEnv.ts` lets an agent play some seats while the built-in AI plays the
rest. The agent is a commander: at each decision point it sees the orders
the built-in AI is considering (defend, invade or claim a star, each with
the AI's own score) and answers with

- `orders`: a score per order key. Orders it leaves out are dropped, and
  higher scores are served first. Leave it out to keep the AI's choices.
- `config`: built-in AI tuning for that seat (`AiConfig`), for example the
  spending split or the invasion margin.
- `research`: the technology to research now and next.

The built-in AI's executor then carries the orders out during the tick
(ship assignment, carrier purchases, waypoints). A seat with no decision
plays exactly like the built-in AI; the specs check this tick for tick.
Decision points are the start of each cycle, a new attack on one of the
seat's stars, and at least every 4 ticks (`maxGap`).

`server.ts` serves the environment as JSON lines on stdin and stdout, and
`python/solaris_env.py` is a standard-library Python client for it:

```sh
python3 sim/python/example_agent.py 1 200   # seed 1, stop at tick 200
```

The executor currently reads the full game state, so it knows true enemy
ship counts when sizing an attack. A fog-of-war executor is future work.

## Known gaps

- Enemy credits, research progress and hidden carriers are unknown and set
  to zero or left out.
- A live view shows enemy stars out of scanning range as of the previous
  tick; the sim treats them as current.
- Object ids are time-based, so two seeded runs match in every value except
  their ids.
