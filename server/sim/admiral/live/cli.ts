#!/usr/bin/env node
// CLI entry point for the Live Human-in-the-Loop Diplomatic Admiral daemon.
//
// Example (Docker read-only advisor):
//   SOLARIS_API=http://localhost:3000 SOLARIS_EMAIL=alpha@test.local SOLARIS_PASSWORD=secret \
//     npx tsx sim/admiral/live/cli.ts --game <gameId> --mode advisor --state data/live_alpha.db
//
// Example (Autonomous with checkpoint & 150cr cash reserve):
//   SOLARIS_API=http://localhost:3000 SOLARIS_EMAIL=alpha@test.local SOLARIS_PASSWORD=secret \
//     npx tsx sim/admiral/live/cli.ts --game <gameId> --mode autonomous \
//       --checkpoint data/admiral3/ema_490.pt --cash-reserve 150 --rps 2 --state data/live_alpha.db

import path from "path";
import readline from "readline";
import { LiveAdmiralDaemon } from "./daemon";
import {
    LiveAction,
    LiveDaemonConfig,
    NeutralBorderPosture,
    OperatingMode,
} from "./types";

function parseArgs(argv: string[]) {
    const out: Record<string, string | boolean> = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith("--")) continue;
        const key = arg.slice(2);
        const next = argv[i + 1];
        if (!next || next.startsWith("--")) {
            out[key] = true;
        } else {
            out[key] = next;
            i++;
        }
    }
    return out;
}

function formatLiveAction(a: LiveAction): string {
    switch (a.type) {
        case "SetResearchNow":
            return `SetResearchNow ${a.preference}${a.reason ? ` (${a.reason})` : ""}`;
        case "SetResearchQueue":
            return `SetResearchQueue [${a.queue.join(", ")}]`;
        case "HireStarSpecialist":
            return `HireStarSpecialist ${a.specialistName ? `'${a.specialistName}'` : `id=${a.specialistId}`} @ ${a.starName ?? a.starId}`;
        case "UpgradeStar":
            return `UpgradeStar ${a.starName ?? a.starId} (${a.infrastructure}${a.expectedCost != null ? `, ${a.expectedCost}cr` : ""})`;
        case "BuildWarpGate":
            return `BuildWarpGate ${a.starName ?? a.starId}${a.expectedCost != null ? ` (${a.expectedCost}cr)` : ""}`;
        case "BuildCarrier":
            return `BuildCarrier @ ${a.starName ?? a.starId} (${a.ships} ships)`;
        case "TransferShips":
            return `TransferShips ${a.carrierName ? `'${a.carrierName}'` : `carrier=${a.carrierId}`} @ ${a.starName ?? a.starId} (carrier=${a.carrierShips}, star=${a.starShips})`;
        case "HireCarrierSpecialist":
            return `HireCarrierSpecialist ${a.specialistName ? `'${a.specialistName}'` : `id=${a.specialistId}`} on ${a.carrierName ? `'${a.carrierName}'` : `carrier=${a.carrierId}`}`;
        case "SaveWaypoints": {
            const hops = (
                a.destinationNames &&
                a.destinationNames.length === a.waypoints.length
                    ? a.destinationNames
                    : a.waypoints.map((w) => w.destination)
            ).join(" -> ");
            return `SaveWaypoints ${a.carrierName ? `'${a.carrierName}'` : `carrier=${a.carrierId}`} [${hops || "CLEAR"}]${a.reason ? ` (${a.reason})` : ""}`;
        }
    }
}

async function promptApproval(
    actions: LiveAction[],
    tick: number,
): Promise<boolean> {
    console.log(`\n[Tick ${tick}] Proposed ${actions.length} LiveAction(s):`);
    actions.forEach((a, idx) => {
        console.log(`  ${idx + 1}. ${formatLiveAction(a)}`);
    });
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    return new Promise((resolve) => {
        rl.question("Approve and dispatch batch? [y/N]: ", (ans) => {
            rl.close();
            resolve(ans.trim().toLowerCase() === "y");
        });
    });
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || !args.game) {
        console.log(
            [
                "Usage: npx tsx sim/admiral/live/cli.ts --game <gameId> [options]",
                "",
                "Options:",
                "  --game <id>                 Solaris game ID (required)",
                "  --api <url>                 API base URL (default: $SOLARIS_API or https://api.solaris.games)",
                "  --mode <mode>               paused | advisor | approval | autonomous (default: advisor)",
                "  --checkpoint <path>         AdmiralNet checkpoint (.pt); uses iter-0 priors if omitted",
                "  --state <path>              SQLite (.db) or JSON (.json) persistence file",
                "  --cash-reserve <n>          Protected credit reserve for manual GUI trades (default: 100)",
                "  --specialist-reserve <n>    Protected specialist tokens for human/battle-swing use (default: 0)",
                "  --neutral-border <m>        defend | open (default: defend)",
                "  --attack <list>             Comma-separated covert targets (e.g. 'Alias' or 'Alias@120')",
                "  --expect-tech <list>        Comma-separated incoming promised techs to skip researching",
                "  --rps <n>                   Max requests/second (default: 3; use 2 for Two-Hybrid Docker)",
                "  --grace-ticks <n>           Idle grace ticks for human-created carriers (default: 2)",
                "  --search                    Enable 1-cycle fog-legal lookahead at cycle start",
                "  --once                      Run a single synchronization/planning pass and exit",
            ].join("\n"),
        );
        process.exit(args.help ? 0 : 1);
    }

    const gameId = String(args.game);
    const mode = String(args.mode ?? "advisor") as OperatingMode;
    const statePath = String(
        args.state ??
            path.join(process.cwd(), "data", "live", `admiral_${gameId}.db`),
    );

    const cfg: LiveDaemonConfig = {
        apiUrl: String(
            args.api ?? process.env.SOLARIS_API ?? "https://api.solaris.games",
        ),
        gameId,
        email: process.env.SOLARIS_EMAIL,
        password: process.env.SOLARIS_PASSWORD,
        cookie: process.env.SOLARIS_COOKIE,
        mode,
        cashReserveCredits: Number(args["cash-reserve"] ?? 100),
        specialistReserveTokens: Number(args["specialist-reserve"] ?? 0),
        allowEmergencySpecialistDip:
            args["no-emergency-specialist-dip"] !== true,
        neutralBorderPosture: String(
            args["neutral-border"] ?? "defend",
        ) as NeutralBorderPosture,
        attackOverrides: args.attack
            ? String(args.attack)
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean)
            : [],
        expectTech: args["expect-tech"]
            ? String(args["expect-tech"])
                  .split(",")
                  .map((s) => s.trim())
                  .filter(Boolean)
            : [],
        maxRequestsPerSecond: Number(args.rps ?? 3),
        carrierGraceTicks: Number(args["grace-ticks"] ?? 2),
        allowEmergencyDefenseReserveDip: args["no-emergency-dip"] !== true,
        tickOffsetMs: Number(args["tick-offset-ms"] ?? 3000),
        pollIntervalMs: Number(args["poll-ms"] ?? 3000),
        deadlineSafetyMs: Number(args["deadline-safety-ms"] ?? 5000),
        statePath,
        checkpointPath: args.checkpoint ? String(args.checkpoint) : undefined,
        useSearchPlanner: Boolean(args.search),
        backfillMaxTicks: Number(args["backfill-ticks"] ?? 24),
        enableSockets: !args.once,
        approvalHandler: mode === "approval" ? promptApproval : undefined,
        onAlert: (alert) => {
            console.log(
                `[${alert.severity.toUpperCase()}][t${alert.tick}][${alert.code}] ${alert.message}`,
            );
        },
        onCycleReport: args.once
            ? undefined
            : (summary) => {
                  const hdr = `[Tick ${summary.tick}][${summary.mode}] Planned=${summary.plannedActions.length} Dispatched=${summary.dispatchedCount}`;
                  console.log(hdr);
                  if (
                      summary.mode === "advisor" ||
                      summary.plannedActions.length > 0
                  ) {
                      for (const a of summary.plannedActions) {
                          console.log(`  - ${formatLiveAction(a)}`);
                      }
                  }
              },
    };

    const daemon = new LiveAdmiralDaemon(cfg);
    const shutdown = () => {
        console.log("Stopping LiveAdmiralDaemon...");
        daemon.stop();
        process.exit(0);
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);

    const info = await daemon.initialize();
    console.log(
        `Initialized LiveAdmiralDaemon as '${info.alias}' (${info.playerId}) on game ${gameId} at tick ${info.tick} [mode=${mode}, rps=${cfg.maxRequestsPerSecond}, reserve=${cfg.cashReserveCredits}cr].`,
    );

    if (args.once) {
        const report = await daemon.runOnce("cli_once");
        console.log(
            `[Tick ${report.tick}][${report.mode}] Planned=${report.plannedActions.length} Dispatched=${report.dispatchedCount}`,
        );
        for (const a of report.plannedActions) {
            console.log(`  - ${formatLiveAction(a)}`);
        }
        daemon.stop();
        return;
    }

    await daemon.start();
}

main().catch((err) => {
    console.error("LiveAdmiralDaemon fatal error:", err);
    process.exit(1);
});
