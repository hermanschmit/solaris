// Command line entry point for the simulator.
//
//   npx tsx sim/cli.ts bench [settings] [ticks]
//       Create an all-AI game from an official settings file
//       (default "standard") and time how fast it ticks.
//
//   npx tsx sim/cli.ts play [settings] [maxTicks] [seed] [replay.json]
//       Play an all-AI game until someone wins or maxTicks (default 3000)
//       is reached, printing standings every 5 cycles. With a file name,
//       also record every tick for sim/viewer.html.
//
//   npx tsx sim/cli.ts step <snapshot.json> [ticks] [seed]
//       Fill in the hidden parts of a saved galaxy view, run it forward with
//       the built-in AI playing every opponent, and print each player's
//       stars and ships.
import "./env";
import fs from "fs";
import { SimEngine } from "./engine";
import { determinize } from "./snapshot";
import { ReplayRecorder } from "./replay";
import { Game } from "../services/types/Game";

const report = (game: Game) => {
    for (const p of game.galaxy.players) {
        const pid = p._id.toString();
        const stars = game.galaxy.stars.filter(
            (s) => s.ownedByPlayerId?.toString() === pid,
        );
        const carriers = game.galaxy.carriers.filter(
            (c) => c.ownedByPlayerId?.toString() === pid,
        );
        const ships =
            stars.reduce((a, s) => a + (s.ships ?? 0), 0) +
            carriers.reduce((a, c) => a + (c.ships ?? 0), 0);

        console.log(
            `${p.alias.padEnd(16)} stars=${String(stars.length).padStart(3)} ` +
                `carriers=${String(carriers.length).padStart(3)} ships=${String(ships).padStart(5)}` +
                (p.defeated ? " (defeated)" : ""),
        );
    }
};

const bench = async (settingsName = "standard", ticks = 100) => {
    const engine = new SimEngine();
    engine.seed(1);

    let t = Date.now();
    const game = await engine.createGame(engine.loadSettings(settingsName));
    console.log(
        `Created ${game.galaxy.players.length} players, ${game.galaxy.stars.length} stars in ${Date.now() - t}ms`,
    );

    t = Date.now();
    await engine.tick(game, ticks);
    const ms = Date.now() - t;
    console.log(
        `Ticked to ${game.state.tick} in ${ms}ms (${(ms / ticks).toFixed(1)}ms per tick, all players on built-in AI)`,
    );

    report(game);
};

const play = async (
    settingsName = "standard",
    maxTicks = 3000,
    seed = 1,
    replayFile?: string,
) => {
    const engine = new SimEngine();
    engine.seed(seed);

    const game = await engine.createGame(engine.loadSettings(settingsName));
    const cycleTicks = game.settings.galaxy.productionTicks;
    const recorder = replayFile
        ? new ReplayRecorder(game, settingsName, seed)
        : null;
    const t = Date.now();

    recorder?.capture(game);

    while (!game.state.endDate && game.state.tick < maxTicks) {
        for (
            let i = 0;
            i < cycleTicks * 5 &&
            !game.state.endDate &&
            game.state.tick < maxTicks;
            i++
        ) {
            await engine.tick(game, 1);
            recorder?.capture(game);
        }
        console.log(
            `\nTick ${game.state.tick} (cycle ${Math.floor(game.state.tick / cycleTicks)}), ` +
                `${((Date.now() - t) / 1000).toFixed(1)}s elapsed`,
        );
        report(game);
    }

    const winner = game.state.winner
        ? game.galaxy.players.find(
              (p) => p._id.toString() === game.state.winner!.toString(),
          )?.alias
        : null;
    console.log(
        `\n${winner ? `${winner} won` : "No winner"} at tick ${game.state.tick} ` +
            `in ${((Date.now() - t) / 1000).toFixed(1)}s (seed ${seed})`,
    );

    if (recorder && replayFile) {
        recorder.save(replayFile);
        console.log(`Saved replay to ${replayFile}`);
    }
};

const step = async (file: string, ticks = 20, seed = 1) => {
    const engine = new SimEngine();
    engine.seed(seed);

    const snapshot = JSON.parse(fs.readFileSync(file, "utf8"));
    const game = determinize(engine, snapshot, { seed });
    const start = game.state.tick;

    const t = Date.now();
    await engine.tick(game, ticks);
    console.log(
        `Tick ${start} -> ${game.state.tick} in ${Date.now() - t}ms (seed ${seed})`,
    );

    report(game);
};

const [command, ...args] = process.argv.slice(2);

const run = async () => {
    switch (command) {
        case "bench":
            return bench(args[0], args[1] ? Number(args[1]) : undefined);
        case "play":
            return play(
                args[0],
                args[1] ? Number(args[1]) : undefined,
                args[2] ? Number(args[2]) : undefined,
                args[3],
            );
        case "step":
            return step(
                args[0],
                args[1] ? Number(args[1]) : undefined,
                args[2] ? Number(args[2]) : undefined,
            );
        default:
            console.log(
                "Usage: cli.ts bench [settings] [ticks] | play [settings] [maxTicks] [seed] [replay.json] | step <snapshot.json> [ticks] [seed]",
            );
    }
};

run()
    .then(() => process.exit(0))
    .catch((e) => {
        console.error(e);
        process.exit(1);
    });
