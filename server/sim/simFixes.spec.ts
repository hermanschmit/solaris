import "./env";
import { SimEngine } from "./engine";
import { AgentEnv } from "./agentEnv";
import { AdmiralEnv } from "./admiral/admiralEnv";

describe("SimEngine & Env lookahead/determinism fixes (#46, #50, #51, #52)", () => {
    it("produces identical game._id and entity ids across runs with the same seed (#52)", async () => {
        const engine = new SimEngine();

        engine.seed(7);
        const g1 = await engine.createGame(engine.loadSettings("standard"));

        engine.seed(7);
        const g2 = await engine.createGame(engine.loadSettings("standard"));

        expect(g1._id.toString()).toBe(g2._id.toString());
        expect(g1._id.toString()).toMatch(/^6000000000000007/);
        expect(g1.galaxy.players[0]._id.toString()).toBe(
            g2.galaxy.players[0]._id.toString(),
        );
        expect(g1.galaxy.stars[0]._id.toString()).toBe(
            g2.galaxy.stars[0]._id.toString(),
        );
    });

    it("restores ObjectId generator after lookahead and mints disjoint ids in rollouts (#46)", async () => {
        // Run A: baseline without lookahead
        const envA = new AgentEnv();
        await envA.reset({
            agents: [0],
            settings: "standard",
            seed: 1,
            maxTicks: 120,
            fog: true,
        });
        while (envA.game.state.tick < 41) {
            await envA.engine.tick(envA.game, 1);
        }
        while (envA.game.state.tick < 65) {
            await envA.engine.tick(envA.game, 1);
        }
        const starsA = envA.standings().players.map((p) => p.stars);

        // Run B: identical game with a 4-tick lookahead at tick 41
        const envB = new AgentEnv();
        await envB.reset({
            agents: [0],
            settings: "standard",
            seed: 1,
            maxTicks: 120,
            fog: true,
        });
        while (envB.game.state.tick < 41) {
            await envB.engine.tick(envB.game, 1);
        }
        await envB.lookahead(0, [{}, {}], 4, 1);
        while (envB.game.state.tick < 65) {
            await envB.engine.tick(envB.game, 1);
        }
        const starsB = envB.standings().players.map((p) => p.stars);

        // Lookahead must not alter the main game's RNG/ObjectId sequence or outcome!
        expect(starsB).toEqual(starsA);
    }, 30_000);

    it("isolates AIService._checkInCycle per Game clone (#50)", async () => {
        const engine = new SimEngine();
        engine.seed(11);
        const game = await engine.createGame(engine.loadSettings("standard"));
        const copy = engine.clone(game);
        const ai = engine.container.aiService;

        // Same game._id, distinct Game objects: WeakMap must isolate them.
        expect(copy._id.toString()).toBe(game._id.toString());
        const mapOrig = new Map([[game.galaxy.players[0]._id.toString(), 1]]);
        ai._checkInCycle.set(game, mapOrig);
        expect(ai._checkInCycle.get(copy)).toBeUndefined();
    });

    it("remaps courting and givenUp to copy players during lookahead without mutating originals (#51)", async () => {
        const env = new AdmiralEnv();
        await env.reset({
            admiralSeats: [0],
            settings: "standard",
            seed: 9,
            maxTicks: 80,
            fog: true,
        });
        const seat = (env as any).seats.get(0);
        const origTarget = env.game.galaxy.players[1];
        const origGivenUp = env.game.galaxy.players[2];
        seat.courting = origTarget;
        seat.givenUp = new Set([origGivenUp]);

        const world = {
            ships: env.game.galaxy.stars.map((s) => s.ships ?? 0),
        };
        await env.lookaheadAdmiral(0, [{}], [world], 2, 1);

        // After lookahead, seat.courting and seat.givenUp still point to original game players.
        expect(seat.courting).toBe(origTarget);
        expect(seat.givenUp.has(origGivenUp)).toBeTrue();
        expect(seat.givenUp.size).toBe(1);
    }, 20_000);
});
