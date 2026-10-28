import mongoose from "mongoose";
import { SimEngine } from "../sim/engine";
import { determinize } from "../sim/snapshot";
import { AgentEnv } from "../sim/agentEnv";
import { BeliefRecorder, HIDDEN, VISIBLE } from "../sim/beliefData";
import { SeatMemory, applyFog } from "../sim/fog";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";

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
    describe("agent environment", () => {
        const play = async (
            env: AgentEnv,
            ticks: number,
            decide: (seat: number) => any,
        ) => {
            let result = await env.reset({
                seed: 4,
                agents: [9],
                maxTicks: ticks,
            });
            while (!result.done) {
                for (const seat of result.seats) {
                    await env.observe(seat);
                    env.act(seat, decide(seat));
                }
                result = await env.step();
            }
            return env.game;
        };

        it("plays exactly like the built-in AI when the agent keeps its choices", async () => {
            const plain = await newGame(4);
            await engine.tick(plain, 60);

            const env = new AgentEnv();
            const game = await play(env, 60, () => ({}));

            expect(game.state.tick).toBe(60);
            expect(fingerprint(env.engine, game)).toEqual(
                fingerprint(engine, plain),
            );
        });

        it("acts only on the orders the agent scores", async () => {
            const env = new AgentEnv();
            const result = await env.reset({ seed: 4, agents: [9] });
            const observation = await env.observe(9);

            expect(result.seats).toEqual([9]);
            expect(observation.candidates.length).toBeGreaterThan(0);
            expect(observation.candidates[0].key).toMatch(
                /^(defend|invade|claim):/,
            );

            // Dropping every order means never claiming a star.
            const game = await play(env, 60, () => ({
                orders: {},
                research: "weapons",
            }));
            const agent = game.galaxy.players[9];
            const owned = (p: any) =>
                game.galaxy.stars.filter(
                    (s) => s.ownedByPlayerId?.toString() === p._id.toString(),
                ).length;

            expect(agent.researchingNow).toBe("weapons");
            expect(owned(agent)).toBeLessThan(
                Math.max(...game.galaxy.players.slice(0, 9).map(owned)),
            );
            expect(owned(agent)).toBeLessThanOrEqual(6);
        });
    });
    describe("belief data", () => {
        it("records what each player sees as the galaxy view shows it", async () => {
            const game = await newGame(4);
            await engine.tick(game, 40);

            const recorder = new BeliefRecorder(engine, game, { seed: 4 });
            recorder.capture(game);

            const seat = 3;
            const player = game.galaxy.players[seat];
            engine.setExternallyControlled(
                game,
                player._id,
                new mongoose.Types.ObjectId(),
            );
            const view = await engine.viewAs(game, player.userId!);
            const stars = game.galaxy.stars.length;
            const visibility = recorder.visibility[0].slice(
                seat * stars,
                (seat + 1) * stars,
            );

            const inRange = new Set(
                view.galaxy.stars
                    .filter((s: any) => s.isInScanningRange)
                    .map((s: any) => s._id.toString()),
            );
            game.galaxy.stars.forEach((s, i) =>
                expect(visibility[i] !== HIDDEN).toBe(
                    inRange.has(s._id.toString()),
                ),
            );
            expect(
                visibility.filter((v) => v === VISIBLE).length,
            ).toBeGreaterThan(0);
            expect(inRange.size).toBeLessThan(stars);

            // Carriers the view shows are the ones marked visible to the seat.
            const shown = new Set(
                view.galaxy.carriers.map((c: any) => c._id.toString()),
            );
            const fields = recorder.header.carrierFields;
            const rows = recorder.carriers[0];
            game.galaxy.carriers.forEach((c, i) => {
                const mask =
                    rows[i * fields.length + fields.indexOf("visible")];
                expect(Boolean(mask & (1 << seat))).toBe(
                    shown.has(c._id.toString()),
                );
            });

            const file = path.join(os.tmpdir(), `belief-${process.pid}.sbd`);
            recorder.save(file);
            const raw = zlib.gunzipSync(fs.readFileSync(file));
            fs.unlinkSync(file);
            const header = JSON.parse(
                raw.subarray(8, 8 + raw.readUInt32LE(4)).toString(),
            );

            expect(raw.subarray(0, 4).toString()).toBe("SBD1");
            expect(header.arrays.map((a: any) => a.name)).toContain(
                "visibility",
            );
            expect(
                header.arrays.find((a: any) => a.name === "stars").shape,
            ).toEqual([1, stars, header.starFields.length]);
        });
    });
    describe("fog of war for agents", () => {
        it("shows the executor only what the seat knows, then puts it back", async () => {
            const game = await newGame(4);
            await engine.tick(game, 60);

            const player = game.galaxy.players[3];
            const id = player._id.toString();
            const memory = new SeatMemory(game.galaxy.stars.length);
            memory.update(engine, game, player);
            const before = fingerprint(engine, game);
            const carriers = game.galaxy.carriers.length;

            const restore = applyFog(game, player, memory, (i) =>
                i === 0 ? 123 : undefined,
            );

            expect(game.galaxy.carriers.length).toBeLessThan(carriers);
            for (const c of game.galaxy.carriers) {
                const own = c.ownedByPlayerId!.toString() === id;
                expect(own || memory.view!.carriers.has(c)).toBe(true);
                if (!own) expect(c.waypoints.length).toBeLessThanOrEqual(1);
            }
            game.galaxy.stars.forEach((s, i) => {
                if (s.ownedByPlayerId?.toString() === id) return;
                if (memory.view!.stars[i] === VISIBLE) return;
                expect(s.shipsActual).toBe(i === 0 ? 123 : memory.ships[i]);
                if (
                    memory.view!.stars[i] === HIDDEN &&
                    memory.seenTick[i] < 0
                ) {
                    expect(s.infrastructure.industry).toBe(0);
                }
            });

            restore();
            expect(fingerprint(engine, game)).toEqual(before);
            expect(game.galaxy.carriers.length).toBe(carriers);
        });

        it("plays agent seats under fog and reports what they see", async () => {
            const env = new AgentEnv();
            let result = await env.reset({
                seed: 4,
                agents: [9],
                fog: true,
                maxTicks: 60,
            });
            while (!result.done) {
                for (const seat of result.seats) {
                    await env.observe(seat);
                    env.act(seat, {});
                }
                result = await env.step();
            }

            const { frame } = await env.observe(9, { frame: true });
            const stars = env.game.galaxy.stars.length;
            const owned = env.standings().players[9].stars;

            expect(env.game.state.tick).toBe(60);
            expect(owned).toBeGreaterThan(1);
            expect(frame.visibility.length).toBe(stars);
            expect(
                frame.visibility.filter((v: number) => v !== HIDDEN).length,
            ).toBeLessThan(stars);
            expect(frame.carriers.owner.length).toBeGreaterThan(0);
            expect(env.header().stars.length).toBe(stars);
        });
    });
});
