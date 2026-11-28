import "./env";
import mongoose from "mongoose";
import { SimEngine } from "./engine";
import { determinize } from "./snapshot";

describe("snapshot.determinize fixes (#47, #48, #49)", () => {
    it("refreshes stale viewer lastSeen so viewer is not marked AFK/defeated on tick 1 (#47)", async () => {
        const engine = new SimEngine();
        engine.seed(5);
        const game = await engine.createGame(engine.loadSettings("standard"));
        const raw = engine.toJSON(game);
        const viewer = raw.galaxy.players[0];
        viewer.userId = new mongoose.Types.ObjectId().toString();
        viewer.lastSeen = new Date("2020-01-01T00:00:00Z").toISOString();
        viewer.afk = false;
        viewer.defeated = false;

        const det = determinize(engine, raw, { seed: 1 });
        await engine.tick(det, 1);

        expect(det.galaxy.players[0].afk).toBeFalse();
        expect(det.galaxy.players[0].defeated).toBeFalse();
    });

    it("estimates non-zero hidden opponent credits and research progress per seed (#48)", async () => {
        const engine = new SimEngine();
        engine.seed(6);
        const game = await engine.createGame(engine.loadSettings("standard"));
        const raw = engine.toJSON(game);
        raw.galaxy.players[0].userId = new mongoose.Types.ObjectId().toString();
        for (let i = 1; i < raw.galaxy.players.length; i++) {
            const p = raw.galaxy.players[i];
            delete p.credits;
            delete p.creditsSpecialists;
            p.stats = {
                totalEconomy: 20,
                totalIndustry: 15,
                totalScience: 10,
                totalShips: 100,
            };
            for (const tech of Object.keys(p.research ?? {})) {
                delete p.research[tech].progress;
            }
        }

        const det = determinize(engine, raw, { seed: 42 });
        const opp = det.galaxy.players[1];
        expect(opp.credits).toBeGreaterThan(0);
        expect(
            opp.research[opp.researchingNow].progress,
        ).toBeGreaterThanOrEqual(0);

        const detZero = determinize(engine, raw, {
            seed: 42,
            estimateHiddenResources: false,
        });
        expect(detZero.galaxy.players[1].credits).toBe(0);
    });

    it("assigns distinct placeholder userIds when opponentsUseBuiltInAI is false (#49)", async () => {
        const engine = new SimEngine();
        engine.seed(7);
        const game = await engine.createGame(engine.loadSettings("standard"));
        const raw = engine.toJSON(game);
        const viewerUid = new mongoose.Types.ObjectId().toString();
        raw.galaxy.players[0].userId = viewerUid;

        const det = determinize(engine, raw, {
            seed: 1,
            opponentsUseBuiltInAI: false,
        });
        const uids = det.galaxy.players.map((p) => p.userId?.toString());
        expect(uids[0]).toBe(viewerUid);
        expect(new Set(uids).size).toBe(det.galaxy.players.length);
    });
});
