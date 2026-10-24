// Check the simulator against the real game.
//
//   npx tsx sim/compare.ts <before.json> <after.json> [seed]
//
// Both files are your own views of the same game (from fetchGalaxy.ts or
// DevTools). The earlier one is run forward to the later one's tick and the
// viewer's own stars, carriers and credits are compared, since those are
// the parts the view shows in full.
import "./env";
import fs from "fs";
import { SimEngine } from "./engine";
import { determinize } from "./snapshot";

const id = (v: any) => (v == null ? null : v.toString());

const main = async () => {
    const [beforeFile, afterFile, seedArg] = process.argv.slice(2);
    const before = JSON.parse(fs.readFileSync(beforeFile, "utf8"));
    const after = JSON.parse(fs.readFileSync(afterFile, "utf8"));
    const seed = seedArg ? Number(seedArg) : 1;

    const viewer = before.galaxy.players.find((p) => p.userId != null);
    if (!viewer) throw new Error("The first file is not a player's own view.");
    const vid = id(viewer._id);

    const engine = new SimEngine();
    engine.seed(seed);
    const game = determinize(engine, before, { seed });
    await engine.tick(game, after.state.tick - before.state.tick);

    const sim = engine.toJSON(game);
    let mismatches = 0;
    const check = (what: string, predicted: any, actual: any) => {
        const ok = JSON.stringify(predicted) === JSON.stringify(actual);
        if (!ok) mismatches++;
        console.log(
            `${ok ? "ok  " : "DIFF"} ${what}: predicted ${JSON.stringify(predicted)}, actual ${JSON.stringify(actual)}`,
        );
    };

    console.log(
        `Tick ${before.state.tick} -> ${after.state.tick} as ${viewer.alias}`,
    );

    const simPlayer = sim.galaxy.players.find((p) => id(p._id) === vid);
    const realPlayer = after.galaxy.players.find((p) => id(p._id) === vid);
    check("credits", simPlayer.credits, realPlayer.credits);
    check(
        "research",
        Object.fromEntries(
            Object.entries(simPlayer.research).map(([k, v]: any) => [
                k,
                [v.level, v.progress],
            ]),
        ),
        Object.fromEntries(
            Object.entries(realPlayer.research).map(([k, v]: any) => [
                k,
                [v.level, v.progress],
            ]),
        ),
    );

    const realStars = after.galaxy.stars.filter(
        (s) => id(s.ownedByPlayerId) === vid,
    );
    const simStars = new Map(sim.galaxy.stars.map((s) => [id(s._id), s]));
    for (const s of realStars) {
        const p: any = simStars.get(id(s._id));
        check(
            `star ${s.name}`,
            p && {
                owner: id(p.ownedByPlayerId),
                ships: p.ships,
                infrastructure: p.infrastructure,
            },
            { owner: vid, ships: s.ships, infrastructure: s.infrastructure },
        );
    }
    const simOwned = sim.galaxy.stars.filter(
        (s) => id(s.ownedByPlayerId) === vid,
    ).length;
    check("star count", simOwned, realStars.length);

    const realCarriers = after.galaxy.carriers.filter(
        (c) => id(c.ownedByPlayerId) === vid,
    );
    const simCarriers = new Map(sim.galaxy.carriers.map((c) => [id(c._id), c]));
    for (const c of realCarriers) {
        const p: any = simCarriers.get(id(c._id));
        const loc = (l) =>
            l && [Math.round(l.x * 100) / 100, Math.round(l.y * 100) / 100];
        check(
            `carrier ${c.name}`,
            p && {
                ships: p.ships,
                at: loc(p.location),
                orbiting: id(p.orbiting),
            },
            { ships: c.ships, at: loc(c.location), orbiting: id(c.orbiting) },
        );
    }

    console.log(
        mismatches === 0
            ? "\nAll checks match."
            : `\n${mismatches} mismatches.`,
    );
};

main()
    .then(() => process.exit(0))
    .catch((e) => {
        console.error(e);
        process.exit(1);
    });
