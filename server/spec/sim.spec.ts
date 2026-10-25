import mongoose from "mongoose";
import { SimEngine } from "../sim/engine";
import { determinize } from "../sim/snapshot";

// Object ids embed a timestamp, so compare games by the order in which ids
// first appear rather than by their values.
const fingerprint = (engine: SimEngine, game: any) => {
    const json = engine.toJSON(game);
    const ids = new Map<string, number>();
    const text = JSON.stringify({
        tick: json.state.tick,
        stars: json.galaxy.stars.map((s) => [
            s.ownedByPlayerId,
            s.shipsActual,
            s.infrastructure,
        ]),
        carriers: json.galaxy.carriers.map((c) => [
            c.ownedByPlayerId,
            c.ships,
            c.location,
        ]),
        players: json.galaxy.players.map((p) => [p.credits, p.research]),
    });
    return text.replace(/[0-9a-f]{24}/g, (id) => {
        if (!ids.has(id)) ids.set(id, ids.size);
        return `#${ids.get(id)}`;
    });
};

describe("sim engine", () => {
    let engine: SimEngine;

    beforeAll(() => {
        engine = new SimEngine();
    });

    const newGame = async (seed: number) => {
        engine.seed(seed);
        return engine.createGame(engine.loadSettings("standard"));
    };

    it("creates a started standard game without a database", async () => {
        const game = await newGame(1);

        expect(game.galaxy.players.length).toBe(10);
        expect(game.galaxy.stars.length).toBe(200);
        expect(game.state.startDate).not.toBeNull();
        expect(game.state.tick).toBe(0);
    });

    it("advances ticks with the built-in AI playing", async () => {
        const game = await newGame(1);

        await engine.tick(game, 5);

        expect(game.state.tick).toBe(5);
        expect(game.state.endDate).toBeNull();
        expect(game.galaxy.carriers.length).toBeGreaterThan(10);
    });

    it("is reproducible for the same seed", async () => {
        const a = await newGame(7);
        engine.seed(99);
        await engine.tick(a, 10);

        const b = await newGame(7);
        engine.seed(99);
        await engine.tick(b, 10);

        expect(fingerprint(engine, a)).toEqual(fingerprint(engine, b));
    });

    it("clones a game so that ticking the copy leaves the original alone", async () => {
        const game = await newGame(3);
        const before = fingerprint(engine, game);

        const copy = engine.clone(game);
        await engine.tick(copy, 3);

        expect(fingerprint(engine, game)).toEqual(before);
        expect(copy.state.tick).toBe(3);
    });

    it("runs a determinized fog-of-war view", async () => {
        const game = await newGame(5);
        const userId = new mongoose.Types.ObjectId();
        const viewer = game.galaxy.players[0];
        engine.setExternallyControlled(game, viewer._id, userId);
        await engine.tick(game, 10);

        const view = await engine.viewAs(game, userId);
        const hidden = view.galaxy.stars.filter(
            (s) => s.infrastructure == null,
        );
        expect(hidden.length).toBeGreaterThan(0);

        const sim = determinize(engine, view, { seed: 1 });

        // Each opponent's totals are preserved by the fill-in.
        for (const p of view.galaxy.players) {
            if (p.userId) continue;
            const owned = sim.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === p._id.toString(),
            );
            const economy = owned.reduce(
                (a, s) => a + s.infrastructure.economy!,
                0,
            );
            expect(economy).toBe(p.stats.totalEconomy);
        }

        await engine.tick(sim, 5);
        expect(sim.state.tick).toBe(15);
    });
    describe("built-in AI config", () => {
        afterEach(() => engine.aiConfigs.clear());

        it("plays the same with the defaults set explicitly", async () => {
            const plain = await newGame(3);
            await engine.tick(plain, 45);

            const configured = await newGame(3);
            for (const p of configured.galaxy.players) {
                engine.setAiConfig(configured, p._id, {});
            }
            await engine.tick(configured, 45);

            expect(fingerprint(engine, configured)).toEqual(
                fingerprint(engine, plain),
            );
        });

        it("tunes one player without affecting the defaults", async () => {
            const game = await newGame(3);
            const saver = game.galaxy.players[0];

            engine.setAiConfig(game, saver._id, {
                firstTickSciencePercentage: 0,
                firstTickIndustryPercentage: 0,
                lastTickEconomyPercentage: 0,
                carrierReserveEconomyMultiplier: 1000,
            });
            expect(
                engine.aiConfigs.get(saver._id.toString())!.orderPriorities,
            ).toEqual({ defend: 4, invade: 3, claim: 2 });

            await engine.tick(game, 45);

            // With every spending rule turned off, the first player keeps
            // its credits while the others spend theirs.
            const others = game.galaxy.players.slice(1);
            for (const p of others) {
                expect(saver.credits).toBeGreaterThan(p.credits);
            }
        });
    });
});
