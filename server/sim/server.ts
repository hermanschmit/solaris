// Serves an AgentEnv over stdin and stdout, one JSON message per line, so
// agents written in other languages (see sim/python/) can drive it.
//
//   npx tsx sim/server.ts
//
// Each request is {"id": 1, "cmd": "reset", "args": {...}} and gets back
// {"id": 1, "ok": true, "result": ...} or {"id": 1, "ok": false, "error":
// "..."}. Commands:
//
//   reset     {agents, settings?, seed?, aiConfigs?, maxTicks?, maxGap?, fog?,
//              record?, recordInfo?}
//             -> {tick, done, seats}
//   header    {}                       -> the game's stars, ranges, techs
//   observe   {seat, view?, frame?, truthFrame?} -> Observation
//   act       {seat, decision}         -> null
//   step      {}                       -> {tick, done, seats}
//   standings {}                       -> each seat's stars and ships, winner
//
// Anything else the server code prints goes to stderr, so stdout carries
// only replies.
process.env.LOGGING_LEVEL ||= "silent";
console.log = console.info = (...args: any[]) => console.error(...args);

import readline from "readline";

const write = (message: any) =>
    process.stdout.write(JSON.stringify(message) + "\n");

const main = async () => {
    // Imported after console is redirected, since loading the server's config
    // prints to it.
    require("./env");
    const { AgentEnv } = require("./agentEnv");
    const env = new AgentEnv();

    const commands: Record<string, (args: any) => any> = {
        reset: (args) => env.reset(args),
        observe: (args) =>
            env.observe(args.seat, {
                view: args.view,
                frame: args.frame,
                truthFrame: args.truthFrame,
            }),
        header: () => env.header(),
        act: async (args) => {
            await env.act(args.seat, args.decision ?? {});
            return null;
        },
        step: () => env.step(),
        standings: () => env.standings(),
    };

    const lines = readline.createInterface({ input: process.stdin });

    // Requests are handled one at a time, in order.
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
