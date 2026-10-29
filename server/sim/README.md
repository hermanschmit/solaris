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
30 cycles). The built-in AI is most of that time.

The engine speeds the built-in AI up without changing a single result (the
game state is identical tick for tick): bson caches object id strings, the
AI's attack-sizing combat estimates are remembered, and star-to-star
distances for its reachability graphs are measured once per game. The first
500 ticks of seed 1 went from 103 s to 41 s, so a whole game now takes
about one to two minutes on one core. About half of what remains is the
combat estimates, which simulate a battle once for every candidate number
of ships.

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

By default the executor reads the full game state, as the built-in AI
does. Reset with `fog: true` and agent seats play under fog of war instead
(`fog.ts`). Each seat remembers what it has seen, tick by tick, and while
its executor runs, and while it proposes orders, the game shows only that:

- carriers out of scanning range are hidden, and visible enemy carriers
  show only their next waypoint;
- stars whose ships the seat cannot see show believed ships and the
  infrastructure last seen there.

Believed ships come from the agent (`beliefs.ships` in its decision, one
per star), or else are the last ships the seat saw there. Everything is put
back before the tick goes on, so the game itself is unchanged.

`observe(seat, { frame: true })` returns what the seat sees and remembers,
and `header()` the game's constant facts. `belief_data.Static.features`
turns these into belief-model inputs with the same code that builds the
training data. `python/belief_agent.py` runs a trained belief model this
way and passes its estimates to the executor:

```sh
python3 sim/python/belief_agent.py belief_model.pt belief 1 9   # or memory, omniscient
```

## Belief-model training data

A belief model guesses what fog of war hides from a player. `beliefBatch.ts`
plays all-AI games in parallel, one per core, and records each one with
`beliefData.ts`: on every tick, the true state of every star, carrier and
player, and which stars and carriers each player has in scanning range, by
the same rules as the game's galaxy view. Each player gets a randomly tuned
built-in AI so the games vary.

```sh
npx tsx sim/beliefBatch.ts data/belief 100 1   # 100 games, seeds 1-100
python3 sim/python/belief_data.py data/belief belief.npz
python3 sim/python/belief_check.py data/belief
```

A game takes 2 to 4 minutes per core and about 1 MB on disk. Recording adds
about 6% to the run time.

`python/belief_data.py` turns the recordings into samples: one player's view
at one tick, as features per star and per player, with the hidden truth as
targets. The player remembers what it saw: a star out of range keeps the
ships and infrastructure it last had in range, with their age. Frontier
features give each star's travel time to the player's own and to enemy
stars, and the enemy ships the player knows of that could reach it within
one or two cycles. Reach follows each enemy's own hyperspace range, hop by
hop, since tech levels are public. Targets include the true ships and
infrastructure everywhere, the true threat, the enemy ships that actually
arrive within the next cycle, and each enemy's credits and research.
Building features in Python keeps them easy to change without re-running
games. `belief_check.py` prints how much fog hides and the scores of simple
guesses, which a belief model has to beat.

`python/belief_train.py` trains a first belief model with PyTorch, on the
GPU when there is one:

```sh
python3 sim/python/belief_train.py belief.npz 20 belief_model.pt
```

Stars attend to each other with a learned bias by distance, and each star's
input includes its owner's public stats and tech. The model predicts
corrections to what the player already knows, so it starts at the simple
baselines. It reports, on games held out by seed, how well it predicts
attacks on the player's stars in the next cycle, the threat to them, ships
at hidden enemy stars and enemy research, each next to a baseline.

## The commander

`python/commander.py` is the policy network for agent seats. On top of a
belief model's star encoding, it decides a weight for each candidate order
(`orderWeights`, which multiply the built-in AI's own scores as the
executor computes them), the seat's tuning (spending split, invasion margin,
logistics stockpile) and, at the start of each cycle, whether to switch
research. Every head starts at zero, so an untrained commander plays
exactly like the built-in AI: weights of 1, default tuning, research left
alone. A separate critic reads a truth frame (`observe(seat, { truthFrame:
true })`), which the policy never sees.

`python/commander_rollout.py` plays games with the commander in a few seats
under fog, sampling its decisions, and records for each one the policy's
inputs, its actions and their log-probabilities, the critic's inputs, and
the rewards (placement, plus a small bonus per cycle for stars held):

```sh
python3 sim/python/commander_rollout.py data/rollouts 20 1 13 3 belief_model.pt
```

### Training it

`python/ppo_train.py` trains the commander with PPO against the built-in
AI. Each iteration plays a batch of games with the current checkpoint,
computes advantages from the critic (GAE), updates the policy and critic,
and saves the next checkpoint. Every 5 iterations it evaluates: the
commander plays a fixed set of seeds with its most likely decisions, and
its seats' placements are compared with the built-in AI's in the same
seats, where the built-in AI sees the whole game.

```sh
# 20 iterations of 52 games on 13 workers, starting from the belief
# model, evaluating on 26 seeds
python3 sim/python/ppo_train.py data/ppo1 20 52 13 belief_model.pt 26
```

`log.jsonl` has a line per iteration (mean placement of the commander's
seats in training games, losses, how far the policy moved) and
`eval.jsonl` a line per evaluation (`placement` against
`builtin_placement`, lower is better, and how many seats did better or
worse). Run it again with the same directory to continue. Set
`PPO_MAX_TICKS` to cut games short when trying it out.

### The league

`python/league_train.py` continues training a commander in a league. Each
game seats the learner in 3 seats, league members (frozen commanders) in 4
and the built-in AI in the other 3. The league starts with the commander
you give it, and the learner joins it as a new member every 10 iterations.
Members the learner places below more often are picked more often.

```sh
# 50 iterations of 52 games on 13 workers, starting from a PPO checkpoint,
# evaluating on 100 seeds against the built-in AI
python3 sim/python/league_train.py data/league1 data/ppo1/ckpt_75.pt 50 52 13 100
```

`league.json` lists the members and, for each, how often the learner has
recently placed above it (`rate`); `log.jsonl` has the same per iteration
under `placed_above`. Evaluation against the built-in AI is as in
`ppo_train.py`.

One game in 4 is also recorded in `data/league1/belief/`, in the format
`beliefBatch.ts` writes, so the belief model can be retrained on games
where commanders play, not only the built-in AI:

```sh
python3 sim/python/belief_data.py data/league1/belief league_belief.npz
python3 sim/python/belief_train.py league_belief.npz 20 belief_league.pt
```

To mix in the original all-AI games, put both sets of `.sbd` files in one
directory first. A retrained belief model starts new commanders
(`ppo_train.py`'s belief argument); a trained commander keeps its own copy.

Two more tools:

```sh
# Average checkpoints trained from the same start ("model soup")
python3 sim/python/average_commanders.py avg.pt data/ppo1/ckpt_60.pt data/ppo1/ckpt_75.pt

# Evaluate any checkpoint against the built-in AI (reusing a run's baseline)
python3 sim/python/commander_eval.py avg.pt 100 13 data/ppo1/eval_baseline.json
```

## Known gaps

- Enemy credits, research progress and hidden carriers are unknown and set
  to zero or left out.
- A live view shows enemy stars out of scanning range as of the previous
  tick; the sim treats them as current.
- Object ids are time-based, so two seeded runs match in every value except
  their ids.
