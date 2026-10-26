// Plays many all-AI games in parallel and records belief-model training
// data for each (see sim/beliefData.ts).
//
//   npx tsx sim/beliefBatch.ts <outDir> [games] [firstSeed] [workers] [settings] [maxTicks]
//
// Game n uses seed firstSeed + n and writes <outDir>/g<seed>.sbd. Each
// player gets a randomly tuned built-in AI, so players differ in how they
// spend, attack and value stars. Games already in outDir are skipped, so
// an interrupted batch can be resumed.
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { AiConfig } from "../services/types/Ai";
import { seededRng, Rng } from "./random";

const between = (rng: Rng, lo: number, hi: number) => lo + (hi - lo) * rng();

// A random tuning around the defaults. About a quarter of players keep the
// defaults.
export const randomAiConfig = (rng: Rng): Partial<AiConfig> => {
    if (rng() < 0.25) return {};

    const science = Math.round(between(rng, 0, 40));

    return {
        firstTickSciencePercentage: science,
        firstTickIndustryPercentage: Math.round(
            between(rng, 10, Math.min(60, 90 - science)),
        ),
        invasionAttackFactor: between(rng, 1.1, 2.5),
        logisticStockpileCycles: between(rng, 0.3, 1.5),
        carrierReserveEconomyMultiplier: between(rng, 2, 10),
        orderReachCycles: Math.round(between(rng, 1, 4)),
        starValueWeights: {
            economy: between(rng, 0.5, 2),
            industry: between(rng, 1, 4),
            science: between(rng, 1, 5),
        },
        orderPriorities:
            rng() < 0.3
                ? { defend: 4, invade: 2, claim: 3 }
                : { defend: 4, invade: 3, claim: 2 },
    };
};

const playOne = async (
    outDir: string,
    seed: number,
    settingsName: string,
    maxTicks: number,
) => {
    // Loaded here so the parent process does not pay for it.
    require("./env");
    const { SimEngine } = require("./engine");
    const { BeliefRecorder } = require("./beliefData");

    const engine = new SimEngine();
    const rng = seededRng(seed * 7919 + 17);

    engine.seed(seed);
    const game = await engine.createGame(engine.loadSettings(settingsName));

    const aiConfigs = game.galaxy.players.map(() => randomAiConfig(rng));
    game.galaxy.players.forEach((p: any, i: number) =>
        engine.setAiConfig(game, p._id, aiConfigs[i]),
    );

    const recorder = new BeliefRecorder(engine, game, {
        settings: settingsName,
        seed,
        aiConfigs,
    });
    const t = Date.now();

    recorder.capture(game);
    while (!game.state.endDate && game.state.tick < maxTicks) {
        await engine.tick(game, 1);
        recorder.capture(game);
    }

    const file = path.join(outDir, `g${seed}.sbd`);
    recorder.save(file + ".tmp");
    fs.renameSync(file + ".tmp", file);

    return {
        seed,
        ticks: game.state.tick,
        winner: recorder.header.winner,
        seconds: (Date.now() - t) / 1000,
    };
};

// A worker plays the seeds it is given, one after another.
const worker = async (
    outDir: string,
    seeds: number[],
    settingsName: string,
    maxTicks: number,
) => {
    process.env.LOGGING_LEVEL ||= "silent";
    // Stdout carries only results.
    console.log = console.info = (...args: any[]) => console.error(...args);

    for (const seed of seeds) {
        const result = await playOne(outDir, seed, settingsName, maxTicks);
        process.stdout.write(JSON.stringify(result) + "\n");
    }
};

const main = async () => {
    const args = process.argv.slice(2);

    if (args[0] === "--worker") {
        const [, outDir, seeds, settingsName, maxTicks] = args;
        return worker(
            outDir,
            seeds.split(",").map(Number),
            settingsName,
            +maxTicks,
        );
    }

    const [
        outDir,
        games = "10",
        firstSeed = "1",
        workers = String(Math.max(1, os.cpus().length - 1)),
        settingsName = "standard",
        maxTicks = "3000",
    ] = args;

    if (!outDir) {
        console.log(
            "Usage: beliefBatch.ts <outDir> [games] [firstSeed] [workers] [settings] [maxTicks]",
        );
        process.exit(1);
    }

    fs.mkdirSync(outDir, { recursive: true });

    const seeds = Array.from(
        { length: +games },
        (_, i) => +firstSeed + i,
    ).filter((s) => !fs.existsSync(path.join(outDir, `g${s}.sbd`)));
    const n = Math.min(+workers, seeds.length);
    const t = Date.now();
    let done = 0;

    console.log(`${seeds.length} games to play on ${n} workers`);

    await Promise.all(
        Array.from({ length: n }, (_, w) => {
            const mine = seeds.filter((_, i) => i % n === w);

            return new Promise<void>((resolve, reject) => {
                const child = spawn(
                    process.execPath,
                    [
                        ...process.execArgv,
                        __filename,
                        "--worker",
                        outDir,
                        mine.join(","),
                        settingsName,
                        maxTicks,
                    ],
                    { stdio: ["ignore", "pipe", "inherit"] },
                );

                let buffer = "";
                child.stdout.on("data", (chunk) => {
                    buffer += chunk;
                    const lines = buffer.split("\n");
                    buffer = lines.pop()!;

                    for (const line of lines) {
                        if (!line.startsWith("{")) continue;
                        const r = JSON.parse(line);
                        done++;
                        console.log(
                            `[${done}/${seeds.length}] seed ${r.seed}: ${r.ticks} ticks, ` +
                                `winner ${r.winner}, ${r.seconds.toFixed(0)}s ` +
                                `(${((Date.now() - t) / 1000).toFixed(0)}s elapsed)`,
                        );
                    }
                });
                child.on("exit", (code) =>
                    code === 0
                        ? resolve()
                        : reject(new Error(`Worker ${w} exited with ${code}`)),
                );
            });
        }),
    );
};

if (require.main === module) main();
