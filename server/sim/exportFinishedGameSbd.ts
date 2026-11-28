#!/usr/bin/env node
// Exports one or more finished Solaris games (via Time Machine
// `GET /api/game/:gameId/galaxy?tick=T`) into standard `.sbd` (SBD1) files
// readable by `belief_data.py` and `admiral_pretrain.py`.
//
// Once a Solaris game is finished (`state.endDate != null`) and before it is
// cleaned (`!state.cleaned`), Solaris lifts fog of war on Time Machine
// snapshots (`ViewpointKind.Finished`), exposing true star ownership, ships,
// infrastructure, warp gates, specialists, and carriers on every tick.
//
// Usage:
//   SOLARIS_EMAIL=... SOLARIS_PASSWORD=... \
//     npx tsx sim/exportFinishedGameSbd.ts --game <gameId> --out-dir data/human_sbd
//
//   SOLARIS_COOKIE='connect.sid=...' \
//     npx tsx sim/exportFinishedGameSbd.ts --game <id1> --game <id2> --out-dir data/human_sbd --rps 2

import "./env";
import fs from "fs";
import path from "path";
import { SimEngine } from "./engine";
import { BeliefRecorder } from "./beliefData";
import { Game } from "../services/types/Game";
import { SolarisLiveClient } from "./admiral/live/solarisClient";

const strId = (v: any): string | null => (v == null ? null : v.toString());

/**
 * Normalizes a raw finished-game JSON payload from `GET /api/game/:gameId/galaxy`
 * into a `SimEngine` `Game` suitable for `BeliefRecorder.capture()`.
 */
export function hydrateFinishedSnapshot(
    engine: SimEngine,
    rawSnapshot: any,
): Game {
    const json = JSON.parse(JSON.stringify(rawSnapshot));
    const galaxy = json.galaxy;

    for (const star of galaxy.stars ?? []) {
        star.naturalResources ??= { economy: 0, industry: 0, science: 0 };
        star.infrastructure ??= { economy: 0, industry: 0, science: 0 };
        star.infrastructure.economy ??= 0;
        star.infrastructure.industry ??= 0;
        star.infrastructure.science ??= 0;
        star.shipsActual ??= star.ships ?? 0;
        star.ships ??= Math.floor(star.shipsActual);
        star.homeStar ??= (galaxy.players ?? []).some(
            (p: any) => strId(p.homeStarId) === strId(star._id),
        );
        star.specialistId ??= null;
        star.specialistExpireTick ??= null;
        star.warpGate = Boolean(star.warpGate);
    }

    for (const carrier of galaxy.carriers ?? []) {
        carrier.ships ??= 0;
        carrier.waypoints ??= [];
        carrier.specialistId ??= null;
        carrier.specialistExpireTick ??= null;
    }

    const startLevels =
        json.settings?.technology?.startingTechnologyLevel ?? {};
    const techNames = Object.keys(startLevels).filter((k) => /^[^_$]/.test(k));

    for (const player of galaxy.players ?? []) {
        player.research ??= {};
        for (const tech of techNames) {
            player.research[tech] ??= {
                level: Number(startLevels[tech] ?? 1),
                progress: 0,
            };
            player.research[tech].level ??= Number(startLevels[tech] ?? 1);
            player.research[tech].progress ??= 0;
        }
        player.researchingNow ??= "";
        player.researchQueue ??= [];
        player.credits ??= 0;
        player.creditsSpecialists ??= 0;
        player.renownToGive ??= 0;
        player.defeated = Boolean(player.defeated);
    }

    return engine.fromJSON(json);
}

export interface ExportSbdSummary {
    outPath: string;
    ticksRecorded: number;
    firstTick: number;
    lastTick: number;
    players: number;
    stars: number;
    winner: number | null;
}

/**
 * Converts an ordered list of unmasked per-tick galaxy JSON snapshots into a
 * gzipped `.sbd` (`SBD1`) file by recomputing exact per-player fog-of-war
 * visibility masks via `BeliefRecorder` + `scanningService`.
 */
export function exportSnapshotsToSbd(
    engine: SimEngine,
    snapshots: any[],
    outPath: string,
    info: Record<string, any> = {},
): ExportSbdSummary {
    if (!snapshots.length) {
        throw new Error("Cannot export .sbd: no snapshots provided.");
    }

    const sorted = [...snapshots].sort(
        (a, b) => Number(a?.state?.tick ?? 0) - Number(b?.state?.tick ?? 0),
    );
    const firstGame = hydrateFinishedSnapshot(engine, sorted[0]);
    const finalRaw = sorted[sorted.length - 1];

    const recorder = new BeliefRecorder(engine, firstGame, {
        source: "live_finished_game",
        gameId: strId(finalRaw._id),
        gameName: finalRaw?.settings?.general?.name ?? null,
        isAIControlled: (finalRaw?.galaxy?.players ?? []).map((p: any) =>
            Boolean(p.isAIControlled),
        ),
        ...info,
    });

    for (let i = 0; i < sorted.length; i++) {
        const g =
            i === 0 ? firstGame : hydrateFinishedSnapshot(engine, sorted[i]);
        recorder.capture(g);
    }

    fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
    const tmpPath = `${outPath}.tmp`;
    recorder.save(tmpPath);
    fs.renameSync(tmpPath, outPath);

    return {
        outPath,
        ticksRecorded: sorted.length,
        firstTick: Number(sorted[0]?.state?.tick ?? 0),
        lastTick: Number(finalRaw?.state?.tick ?? 0),
        players: firstGame.galaxy.players.length,
        stars: firstGame.galaxy.stars.length,
        winner: recorder.header.winner ?? null,
    };
}

function parseArgs(argv: string[]) {
    const games: string[] = [];
    const opts: Record<string, string | boolean> = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "--game") {
            const next = argv[++i];
            if (next) games.push(next);
            continue;
        }
        if (!arg.startsWith("--")) continue;
        const key = arg.slice(2);
        const next = argv[i + 1];
        if (!next || next.startsWith("--")) {
            opts[key] = true;
        } else {
            opts[key] = next;
            i++;
        }
    }
    return { games, opts };
}

async function fetchAndExportGame(params: {
    client: SolarisLiveClient;
    engine: SimEngine;
    gameId: string;
    outPath: string;
    step: number;
    cacheDir?: string;
    allowUnfinished?: boolean;
}): Promise<ExportSbdSummary> {
    const { client, engine, gameId, outPath, step, cacheDir } = params;
    const latest = await client.getGalaxy(gameId);
    const name = latest?.settings?.general?.name ?? gameId;
    const isFinished = Boolean(latest?.state?.endDate);
    const isCleaned = Boolean(latest?.state?.cleaned);
    const timeMachineEnabled =
        latest?.settings?.general?.timeMachine === "enabled";

    if (!isFinished && !params.allowUnfinished) {
        throw new Error(
            `Game '${name}' (${gameId}) is not finished yet (fog of war is still active). Pass --allow-unfinished only if you intentionally want a partial/fogged export.`,
        );
    }
    if (isCleaned) {
        throw new Error(
            `Game '${name}' (${gameId}) has already been cleaned (Time Machine history was purged).`,
        );
    }
    if (!timeMachineEnabled) {
        throw new Error(
            `Game '${name}' (${gameId}) has Time Machine disabled; historical ticks cannot be fetched.`,
        );
    }

    const minTick = Math.max(
        1,
        Number(latest?.state?.timeMachineMinimumTick ?? 1),
    );
    const maxTick = Number(latest?.state?.tick ?? minTick);
    const ticks: number[] = [];
    for (let t = minTick; t < maxTick; t += step) {
        ticks.push(t);
    }
    if (ticks[ticks.length - 1] !== maxTick) {
        ticks.push(maxTick);
    }

    const gameCacheDir = cacheDir ? path.join(cacheDir, gameId) : null;
    if (gameCacheDir) {
        fs.mkdirSync(gameCacheDir, { recursive: true });
    }

    console.log(
        `Exporting '${name}' (${gameId}): ticks ${minTick}..${maxTick} (${ticks.length} snapshot(s), step=${step})...`,
    );

    const snapshots: any[] = [];
    for (let idx = 0; idx < ticks.length; idx++) {
        const t = ticks[idx];
        const cacheFile = gameCacheDir
            ? path.join(gameCacheDir, `t${t}.json`)
            : null;

        let snap: any;
        if (t === maxTick) {
            snap = latest;
        } else if (cacheFile && fs.existsSync(cacheFile)) {
            snap = JSON.parse(fs.readFileSync(cacheFile, "utf8"));
        } else {
            snap = await client.getGalaxy(gameId, t);
            if (cacheFile) {
                fs.writeFileSync(cacheFile, JSON.stringify(snap));
            }
        }
        snapshots.push(snap);

        if ((idx + 1) % 25 === 0 || idx + 1 === ticks.length) {
            process.stdout.write(
                `\r  fetched ${idx + 1}/${ticks.length} ticks (t=${t})`,
            );
        }
    }
    process.stdout.write("\n");

    const summary = exportSnapshotsToSbd(engine, snapshots, outPath, {
        step,
        exportedAt: new Date().toISOString(),
    });
    console.log(
        `Saved ${summary.outPath} (${summary.ticksRecorded} ticks, ${summary.players} players, ${summary.stars} stars, winner=${summary.winner}).`,
    );
    return summary;
}

async function main() {
    const { games, opts } = parseArgs(process.argv.slice(2));
    if (opts.help || games.length === 0) {
        console.log(
            [
                "Usage: npx tsx sim/exportFinishedGameSbd.ts --game <gameId> [--game <gameId2> ...] [options]",
                "",
                "Auth (via env vars):",
                "  SOLARIS_EMAIL + SOLARIS_PASSWORD   Login via /api/auth/login",
                "  SOLARIS_COOKIE                     Existing session cookie (e.g. connect.sid=...)",
                "",
                "Options:",
                "  --api <url>         API base URL (default: $SOLARIS_API or https://api.solaris.games)",
                "  --game <id>         Finished Solaris game ID (repeatable)",
                "  --out <file.sbd>    Output .sbd path (single game only)",
                "  --out-dir <dir>     Output directory (default: data/human_sbd; writes g_<gameId>.sbd)",
                "  --rps <n>           Max requests/second (default: 2)",
                "  --step <n>          Tick stride (default: 1 = every tick)",
                "  --cache-dir <dir>   Optional directory to cache fetched tick JSONs for resumability",
                "  --allow-unfinished  Allow exporting an in-progress game (not recommended for belief training)",
            ].join("\n"),
        );
        process.exit(opts.help ? 0 : 1);
    }

    if (opts.out && games.length > 1) {
        throw new Error(
            "Use --out-dir (instead of --out) when exporting multiple --game IDs.",
        );
    }

    const apiUrl = String(
        opts.api ?? process.env.SOLARIS_API ?? "https://api.solaris.games",
    );
    const outDir = String(
        opts["out-dir"] ?? path.join(process.cwd(), "data", "human_sbd"),
    );
    const rps = Math.max(0.5, Number(opts.rps ?? 2));
    const step = Math.max(1, Math.floor(Number(opts.step ?? 1)));
    const cacheDir = opts["cache-dir"] ? String(opts["cache-dir"]) : undefined;
    const allowUnfinished = Boolean(opts["allow-unfinished"]);

    const client = new SolarisLiveClient({
        apiUrl,
        email: process.env.SOLARIS_EMAIL,
        password: process.env.SOLARIS_PASSWORD,
        cookie: process.env.SOLARIS_COOKIE,
        maxRequestsPerSecond: rps,
    });
    await client.login();

    const engine = new SimEngine();

    for (const gameId of games) {
        const outPath = opts.out
            ? String(opts.out)
            : path.join(outDir, `g_${gameId}.sbd`);
        await fetchAndExportGame({
            client,
            engine,
            gameId,
            outPath,
            step,
            cacheDir,
            allowUnfinished,
        });
    }
}

if (require.main === module) {
    main().catch((err) => {
        console.error("exportFinishedGameSbd error:", err);
        process.exit(1);
    });
}
