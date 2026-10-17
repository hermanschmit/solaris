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

The built-in AI dominates, so advisor rollouts will want a cheaper opponent
model.

## Known gaps

- Enemy credits, research progress and hidden carriers are unknown and set
  to zero or left out.
- A live view shows enemy stars out of scanning range as of the previous
  tick; the sim treats them as current.
- Object ids are time-based, so two seeded runs match in every value except
  their ids.
