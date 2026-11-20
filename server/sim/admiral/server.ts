// Serves AdmiralEnv over stdin and stdout (JSON-lines RPC), supporting both
// Admiral seats and legacy Commander seats side by side in the same game.
//
//   npx tsx sim/admiral/server.ts
//
// Commands:
//   reset            {admiralSeats?, commanderSeats?, agents?, settings?, seed?,
//                     aiConfigs?, maxTicks?, maxGap?, cycleOnly?, fog?, record?, recordInfo?}
//   header           {}
//   observe          {seat, view?, frame?, truthFrame?}       (Commander seat)
//   observeAdmiral   {seat, truthFrame?}                      (Admiral seat)
//   observeBatch     {items: [{seat, kind?, truthFrame?, frame?, view?}, ...]}
//   act              {seat, decision}                         (Commander seat)
//   actAdmiral       {seat, decision}                         (Admiral seat)
//   actBatch         {items: [{seat, kind?, decision}, ...]}
//   step             {}
//   standings        {}
//   lookahead        {seat, plans, ticks, seed?}
//   lookaheadAdmiral {seat, plans, worlds, ticks, seed?}

process.env.LOGGING_LEVEL ||= "silent";
console.log = console.info = (...args: any[]) => console.error(...args);

import readline from "readline";

const write = (message: any) =>
    process.stdout.write(JSON.stringify(message) + "\n");

process.stdout.on("error", (e: any) => {
    if (e.code === "EPIPE") process.exit(0);
    throw e;
});

const main = async () => {
    require("../env");
    const { AdmiralEnv } = require("./admiralEnv");
    const env = new AdmiralEnv();

    const commands: Record<string, (args: any) => any> = {
        reset: (args) => env.reset(args),
        header: () => env.header(),
        observe: (args) =>
            env.observe(args.seat, {
                view: args.view,
                frame: args.frame,
                truthFrame: args.truthFrame,
            }),
        observeAdmiral: (args) =>
            env.observeAdmiral(args.seat, {
                truthFrame: args.truthFrame,
            }),
        observeBatch: async (args) => {
            const out: any[] = [];
            for (const item of args.items ?? []) {
                const isAdmiral =
                    item.kind === "admiral" ||
                    (item.kind == null && env.admiralSeatSet.has(item.seat));
                if (isAdmiral) {
                    out.push(
                        env.observeAdmiral(item.seat, {
                            truthFrame: item.truthFrame,
                        }),
                    );
                } else {
                    out.push(
                        await env.observe(item.seat, {
                            view: item.view,
                            frame: item.frame,
                            truthFrame: item.truthFrame,
                        }),
                    );
                }
            }
            return out;
        },
        act: async (args) => {
            await env.act(args.seat, args.decision ?? {});
            return null;
        },
        actAdmiral: async (args) => {
            await env.actAdmiral(args.seat, args.decision ?? {});
            return null;
        },
        actBatch: async (args) => {
            for (const item of args.items ?? []) {
                const isAdmiral =
                    item.kind === "admiral" ||
                    (item.kind == null && env.admiralSeatSet.has(item.seat));
                if (isAdmiral) {
                    await env.actAdmiral(item.seat, item.decision ?? {});
                } else {
                    await env.act(item.seat, item.decision ?? {});
                }
            }
            return null;
        },
        step: () => env.step(),
        standings: () => env.standings(),
        lookahead: (args) =>
            env.lookahead(args.seat, args.plans, args.ticks, args.seed),
        lookaheadAdmiral: (args) =>
            env.lookaheadAdmiral(
                args.seat,
                args.plans,
                args.worlds,
                args.ticks,
                args.seed,
            ),
    };

    const lines = readline.createInterface({ input: process.stdin });

    for await (const line of lines) {
        if (!line.trim()) continue;

        let id = null;
        try {
            const request = JSON.parse(line);
            id = request.id ?? null;
            const command = commands[request.cmd];
            if (!command) throw new Error(`Unknown command ${request.cmd}`);
            write({ id, ok: true, result: await command(request.args ?? {}) });
        } catch (e: any) {
            write({ id, ok: false, error: e?.message ?? String(e) });
        }
    }

    process.exit(0);
};

main();
