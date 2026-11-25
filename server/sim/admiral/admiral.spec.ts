import "../env";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import mongoose from "mongoose";
import { SimEngine } from "../engine";
import { HIDDEN } from "../beliefData";
import {
    fastCalculateBasicO1,
    fastEstimateNeeded,
    fastFight,
} from "./fastDefense";
import { TacticalExecutor } from "./tacticalExecutor";
import { AdmiralEnv, BeliefWorld } from "./admiralEnv";

const fingerprint = (engine: SimEngine, game: any) => {
    const json = engine.toJSON(game);
    const ids = new Map<string, number>();
    const text = JSON.stringify({
        tick: json.state.tick,
        stars: json.galaxy.stars.map((s) => [
            s.ownedByPlayerId,
            s.shipsActual,
            s.infrastructure,
            s.warpGate,
        ]),
        carriers: json.galaxy.carriers.map((c) => [
            c.ownedByPlayerId,
            c.ships,
            c.location,
            c.waypoints?.map((w: any) => [w.destination, w.delayTicks]),
        ]),
        players: json.galaxy.players.map((p) => [p.credits, p.research]),
    });
    return text.replace(/[0-9a-f]{24}/g, (id) => {
        if (!ids.has(id)) ids.set(id, ids.size);
        return `#${ids.get(id)}`;
    });
};

describe("Admiral stack", () => {
    describe("fastDefense exactness", () => {
        let rawEngine: SimEngine;

        beforeAll(() => {
            rawEngine = new SimEngine();
        });

        it("matches CombatService.calculateBasic bit-for-bit across ship & weapon sweeps", () => {
            const combat = rawEngine.container.combatService;
            const shipCounts = [0, 1, 2, 3, 7, 15, 16, 49, 128, 999];
            const weapons = [1, 2, 5, 9];

            for (const dShips of shipCounts) {
                for (const aShips of shipCounts) {
                    if (dShips === 0 && aShips === 0) continue;
                    for (const dW of weapons) {
                        for (const aW of weapons) {
                            for (const toStar of [false, true]) {
                                for (const bonus of [false, true]) {
                                    const expected = combat.calculateBasic(
                                        { ships: dShips, weaponsLevel: dW },
                                        { ships: aShips, weaponsLevel: aW },
                                        toStar,
                                        bonus,
                                    );
                                    const actual = fastCalculateBasicO1(
                                        { ships: dShips, weaponsLevel: dW },
                                        { ships: aShips, weaponsLevel: aW },
                                        toStar,
                                        bonus,
                                    );
                                    expect(actual).toEqual(expected);
                                }
                            }
                        }
                    }
                }
            }

            // Verify O(1) handling on huge late-game stacks without timeout.
            const [remD, remA] = fastFight(1_000_000, 7, 1, 950_000, 8);
            expect(remD >= 0 && remA >= 0).toBeTrue();
            expect(remD === 0 || remA === 0).toBeTrue();
        });

        it("matches CombatService.estimateNeeded across 2-group and 3-group star combats", async () => {
            rawEngine.seed(11);
            const game = await rawEngine.createGame(
                rawEngine.loadSettings("standard"),
            );
            const combat = rawEngine.container.combatService;

            const star = game.galaxy.stars.find(
                (s) =>
                    s.ownedByPlayerId?.toString() ===
                    game.galaxy.players[0]._id.toString(),
            )!;
            const p1 = game.galaxy.players[1];
            const p2 = game.galaxy.players[2];
            p1.research.weapons.level = 4;
            p2.research.weapons.level = 6;

            const makeCarrier = (owner: any, ships: number) =>
                ({
                    _id: new mongoose.Types.ObjectId(),
                    ownedByPlayerId: owner._id,
                    name: "Test",
                    orbiting: null,
                    isGift: false,
                    ships,
                    specialistId: null,
                    specialistExpireTick: null,
                    specialist: null,
                    location: { ...star.location },
                    locationNext: { ...star.location },
                    waypoints: [],
                    waypointsLooped: false,
                }) as any;

            for (const defShips of [5, 25, 90]) {
                star.ships = defShips;
                star.shipsActual = defShips;

                // 2-group combat
                const c1 = makeCarrier(p1, 140);
                const r2 = combat.computeStar(game, star, [c1])!;
                const g2 = combat.getDefenderDetailed(r2)!;
                expect(fastEstimateNeeded(r2, g2, "greaterThanZeroShips")).toBe(
                    combat.estimateNeeded(r2, g2, "greaterThanZeroShips"),
                );
                expect(fastEstimateNeeded(r2, g2, "eliminateOtherGroups")).toBe(
                    combat.estimateNeeded(r2, g2, "eliminateOtherGroups"),
                );

                // 3-group multi-faction star combat
                const c2 = makeCarrier(p2, 110);
                const r3 = combat.computeStar(game, star, [c1, c2])!;
                const g3 = combat.getDefenderDetailed(r3)!;
                expect(fastEstimateNeeded(r3, g3, "greaterThanZeroShips")).toBe(
                    combat.estimateNeeded(r3, g3, "greaterThanZeroShips"),
                );
                expect(fastEstimateNeeded(r3, g3, "eliminateOtherGroups")).toBe(
                    combat.estimateNeeded(r3, g3, "eliminateOtherGroups"),
                );
            }
        });
    });

    describe("TacticalExecutor micro", () => {
        it("launches synchronized Time-on-Target (ToT) carriers with exact delayTicks", async () => {
            const env = new AdmiralEnv();
            await env.reset({ admiralSeats: [0], seed: 21, fog: false });
            const game = env.game;
            const p0 = game.galaxy.players[0];
            const p1 = game.galaxy.players[1];
            p0.credits = 2000;
            p0.research.hyperspace.level = 10;
            p0.research.scanning.level = 10;

            // Clear existing carriers and construct a clean 3-star triangle:
            // star 0 (own, close), star 1 (own, farther), star 2 (enemy fortified),
            // all other p0 stars emptied so only stars 0 & 1 participate.
            game.galaxy.carriers = [];
            for (const s of game.galaxy.stars) {
                if (s.ownedByPlayerId?.toString() === p0._id.toString()) {
                    s.ships = 0;
                    s.shipsActual = 0;
                }
            }

            const sClose = game.galaxy.stars[0];
            const sFar = game.galaxy.stars[1];
            const sEnemy = game.galaxy.stars[2];
            const speed = game.settings.specialGalaxy.carrierSpeed;

            sClose.ownedByPlayerId = p0._id;
            sClose.location = { x: 0, y: 0 };
            sClose.ships = 60;
            sClose.shipsActual = 60;
            sClose.warpGate = false;

            sFar.ownedByPlayerId = p0._id;
            sFar.location = { x: -speed * 5, y: 0 };
            sFar.ships = 60;
            sFar.shipsActual = 60;
            sFar.warpGate = false;

            sEnemy.ownedByPlayerId = p1._id;
            sEnemy.location = { x: speed * 2, y: 0 }; // 2 ticks from sClose, 7 ticks from sFar
            sEnemy.ships = 35; // Defender bonus x2 => ~70 base, ~95 w/ 1.35 margin (>60 alone, <=120 combined!)
            sEnemy.shipsActual = 35;
            sEnemy.specialistId = null;
            sEnemy.isAsteroidField = false;
            sEnemy.infrastructure = { economy: 5, industry: 0, science: 0 };
            sEnemy.wormHoleToStarId = null;
            sClose.wormHoleToStarId = null;
            sFar.wormHoleToStarId = null;

            const attackDesire = new Array(game.galaxy.stars.length).fill(0);
            attackDesire[2] = 10.0;
            const garrisonTarget = new Array(game.galaxy.stars.length).fill(0);

            await env.tacticalExecutor.executeTick(game, p0, {
                attackDesire,
                garrisonTarget,
                attackMargin: 1.35,
                maxAttackTicks: 30,
                autoCarrierSpecialist: false,
            });

            const inbound = game.galaxy.carriers.filter(
                (c) =>
                    c.ownedByPlayerId?.toString() === p0._id.toString() &&
                    c.waypoints?.[0]?.destination.toString() ===
                        sEnemy._id.toString(),
            );
            expect(inbound.length).toBe(2);

            const fromClose = inbound.find(
                (c) =>
                    c.waypoints[0].source.toString() === sClose._id.toString(),
            )!;
            const fromFar = inbound.find(
                (c) => c.waypoints[0].source.toString() === sFar._id.toString(),
            )!;
            expect(fromClose).toBeDefined();
            expect(fromFar).toBeDefined();
            expect(fromFar.waypoints[0].delayTicks).toBe(0);
            expect(fromClose.waypoints[0].delayTicks).toBe(5);
        });

        it("freezes upgrades on doomed stars and evacuates garrison before impact", async () => {
            const env = new AdmiralEnv();
            await env.reset({ admiralSeats: [0], seed: 33, fog: false });
            const game = env.game;
            const p0 = game.galaxy.players[0];
            const p1 = game.galaxy.players[1];

            game.state.tick = 1; // First tick of cycle (upgrade window)
            p0.credits = 1500;
            p0.research.hyperspace.level = 6;
            game.galaxy.carriers = [];

            const ownStars = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === p0._id.toString(),
            );
            const doomed = ownStars[0];
            const safe = ownStars[1];
            const speed = game.settings.specialGalaxy.carrierSpeed;

            doomed.location = { x: 0, y: 0 };
            doomed.ships = 30;
            doomed.shipsActual = 30;
            doomed.infrastructure = { economy: 2, industry: 2, science: 1 };

            safe.location = { x: speed * 2, y: 0 };
            safe.ships = 10;
            safe.shipsActual = 10;

            for (let k = 2; k < ownStars.length; k++) {
                ownStars[k].ships = 0;
                ownStars[k].shipsActual = 0;
            }

            // Spawn overwhelming enemy fleet 2 ticks out from `doomed`.
            const enemyStar = game.galaxy.stars.find(
                (s) => s.ownedByPlayerId?.toString() === p1._id.toString(),
            )!;
            enemyStar.location = { x: -speed * 4, y: 0 };
            game.galaxy.carriers.push({
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p1._id,
                name: "Armada",
                orbiting: null,
                isGift: false,
                ships: 500,
                specialistId: null,
                specialistExpireTick: null,
                specialist: null,
                location: { x: -speed * 1.5, y: 0 },
                locationNext: { x: -speed * 0.5, y: 0 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: enemyStar._id,
                        destination: doomed._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            } as any);

            const doomedIdx = game.galaxy.stars.indexOf(doomed);
            const highPrio = new Array(game.galaxy.stars.length).fill(0.01);
            highPrio[doomedIdx] = 1000.0; // Tempt executor to upgrade doomed star!

            await env.tacticalExecutor.executeTick(game, p0, {
                industryPriority: highPrio,
                sciencePriority: highPrio,
                warpGatePriority: highPrio,
                attackDesire: new Array(game.galaxy.stars.length).fill(0),
            });

            // Doomed star must NOT receive upgrades and MUST evacuate its 30 ships.
            expect(doomed.infrastructure.industry).toBe(2);
            expect(doomed.infrastructure.science).toBe(1);
            expect(doomed.warpGate).toBeFalse();

            const evac = game.galaxy.carriers.find(
                (c) =>
                    c.ownedByPlayerId?.toString() === p0._id.toString() &&
                    c.waypoints?.[0]?.source.toString() ===
                        doomed._id.toString(),
            );
            expect(evac).toBeDefined();
            // Leaves 1 rear-guard ship behind on `doomed` for the `1 + B` hit!
            expect(evac!.ships).toBe(29);
            expect(doomed.ships).toBe(1);
        });

        it("enforces $25 carrier gate and stands-and-bleeds when a friendly counter-punch can recapture", async () => {
            const env = new AdmiralEnv();
            await env.reset({ admiralSeats: [0], seed: 34, fog: false });
            const game = env.game;
            const p0 = game.galaxy.players[0];
            const p1 = game.galaxy.players[1];
            const speed = game.settings.specialGalaxy.carrierSpeed;

            p0.credits = 500;
            p0.research.hyperspace.level = 6;
            p0.research.weapons.level = 1;
            p1.research.weapons.level = 1;
            game.galaxy.carriers = [];

            const ownStars = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === p0._id.toString(),
            );
            const doomed = ownStars[0];
            const safe = ownStars[1];
            for (const s of ownStars) {
                s.ships = 0;
                s.shipsActual = 0;
                s.infrastructure = { economy: 0, industry: 0, science: 0 };
            }

            doomed.location = { x: 0, y: 0 };
            safe.location = { x: speed * 4, y: 0 }; // 4 ticks away (> 2 ticks ETA, so cannot reinforce in time!)
            const enemyStar = game.galaxy.stars.find(
                (s) => s.ownedByPlayerId?.toString() === p1._id.toString(),
            )!;
            enemyStar.location = { x: -speed * 4, y: 0 };

            const spawnAttack = (attShips: number) => {
                game.galaxy.carriers = [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        ownedByPlayerId: p1._id,
                        name: "Attacker",
                        orbiting: null,
                        isGift: false,
                        ships: attShips,
                        specialistId: null,
                        specialistExpireTick: null,
                        specialist: null,
                        location: { x: -speed * 1.5, y: 0 }, // 2 ticks until arrival
                        locationNext: { x: -speed * 0.5, y: 0 },
                        waypoints: [
                            {
                                _id: new mongoose.Types.ObjectId(),
                                source: enemyStar._id,
                                destination: doomed._id,
                                action: "nothing",
                                actionShips: 0,
                                delayTicks: 0,
                            },
                        ],
                        waypointsLooped: false,
                    } as any,
                ];
            };

            // Case A: Small stack (6 ships, no orbiting carrier) vs huge 300-ship fleet
            // -> Do NOT waste $25 building an evacuation carrier for < 10 ships.
            doomed.ships = 6;
            doomed.shipsActual = 6;
            safe.ships = 5;
            safe.shipsActual = 5;
            spawnAttack(300);
            const credBefore = p0.credits;
            await env.tacticalExecutor.executeTick(game, p0, {
                attackDesire: new Array(game.galaxy.stars.length).fill(0),
                autoCarrierSpecialist: false,
            });
            expect(p0.credits).toBe(credBefore);
            expect(doomed.ships).toBe(6);

            // Case B: Stand-and-Bleed Counter-Punch!
            // `doomed` has 15 ships (W=1 + B=1 => strike 2 vs attacker W=1 => 2:1 kills!).
            // 35 attackers lose 30 ships, leaving only 5 weakened survivors;
            // `safe` (4 ticks away, within direct jump range) holds 25 ships to immediately recapture!
            doomed.ships = 15;
            doomed.shipsActual = 15;
            safe.ships = 25;
            safe.shipsActual = 25;
            spawnAttack(35);
            await env.tacticalExecutor.executeTick(game, p0, {
                attackDesire: new Array(game.galaxy.stars.length).fill(0),
                autoCarrierSpecialist: false,
            });
            const evacB = game.galaxy.carriers.find(
                (c) =>
                    c.ownedByPlayerId?.toString() === p0._id.toString() &&
                    c.waypoints?.[0]?.source.toString() ===
                        doomed._id.toString(),
            );
            expect(evacB).toBeUndefined();
            expect(doomed.ships).toBe(15);
        });

        it("builds traffic-weighted paired Warp Gates and reroutes eligible in-flight carriers immediately", async () => {
            const env = new AdmiralEnv();
            await env.reset({ admiralSeats: [0], seed: 25, fog: false });
            const game = env.game;
            const p0 = game.galaxy.players[0];
            const p1 = game.galaxy.players[1];
            const ly = game.constants?.distances?.lightYear || 50;

            game.state.tick = 1; // isFirstTick
            p0.research.hyperspace.level = 2; // hsDist = 3 * 50 = 150
            p0.research.terraforming.level = 1;
            game.galaxy.carriers = [];

            const ownStars = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === p0._id.toString(),
            );
            expect(ownStars.length).toBeGreaterThanOrEqual(5);
            const [s0, s1, s2, s3, s4] = ownStars;

            // Move any extra own stars and enemy stars far away so geometry is clean.
            for (const s of game.galaxy.stars) {
                s.warpGate = false;
                s.wormHoleToStarId = null;
                s.ships = 5;
                s.shipsActual = 5;
                s.infrastructure = { economy: 1, industry: 1, science: 1 };
                s.naturalResources = { economy: 25, industry: 25, science: 25 };
                if (s !== s0 && s !== s1 && s !== s2 && s !== s3 && s !== s4) {
                    s.location =
                        s.ownedByPlayerId?.toString() === p0._id.toString()
                            ? { x: 5000, y: 5000 }
                            : { x: -5000, y: -5000 };
                }
            }
            const enemyStar = game.galaxy.stars.find(
                (s) => s.ownedByPlayerId?.toString() === p1._id.toString(),
            )!;
            enemyStar.location = { x: 4.5 * ly, y: 0 };

            // s0 (already gated) <-> s1 is 2.4 LY (120u -> 8t normal vs 3t warped, 5t saved).
            // s2 is a 2-hop ungated detour (s0->s2->s1 = 10t).
            // s4 is isolated far away with high industry & cheap gate cost (old greedy heuristic would pick s4!).
            s0.location = { x: 0, y: 0 };
            s0.warpGate = true;
            s1.location = { x: 2.4 * ly, y: 0 };
            s1.infrastructure.industry = 4;
            s1.naturalResources = { economy: 40, industry: 40, science: 40 };
            s2.location = { x: 1.2 * ly, y: 0.8 * ly };
            s2.naturalResources = { economy: 10, industry: 10, science: 10 };
            s3.location = { x: -0.3 * ly, y: 0 }; // 1-tick hop from s0 -> 0 warp savings!
            s4.location = { x: 40 * ly, y: 40 * ly };
            s4.infrastructure.industry = 20;
            s4.naturalResources = { economy: 50, industry: 50, science: 50 };

            // Fund exactly enough for 1 Warp Gate at s1 (not 2) so the choice is forced.
            const wgCostS1 = (env.tacticalExecutor as any)._warpGateCost(
                game,
                s1,
            ) as number;
            expect(wgCostS1).toBeGreaterThan(0);
            p0.credits = wgCostS1 + 10;

            // In-flight carrier headed s3 -> s0 -> s2 -> s1 (dropAll):
            // once s1 builds a Warp Gate, waypoints[1..] should reroute directly s0 -> s1!
            const inFlightCar: any = {
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p0._id,
                name: "InFlightTransit",
                orbiting: null,
                isGift: false,
                ships: 35,
                specialistId: null,
                specialistExpireTick: null,
                specialist: null,
                location: { x: -0.15 * ly, y: 0 },
                locationNext: { x: 0, y: 0 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s3._id,
                        destination: s0._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s0._id,
                        destination: s2._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s2._id,
                        destination: s1._id,
                        action: "dropAll",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            };

            // Delayed ToT carrier must NOT be rerouted.
            const delayedCar: any = {
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p0._id,
                name: "DelayedToT",
                orbiting: s3._id,
                isGift: false,
                ships: 15,
                specialistId: null,
                specialistExpireTick: null,
                specialist: null,
                location: { ...s3.location },
                locationNext: { ...s3.location },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s3._id,
                        destination: s0._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 2,
                    },
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s0._id,
                        destination: s2._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: s2._id,
                        destination: s1._id,
                        action: "dropAll",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            };
            game.galaxy.carriers = [inFlightCar, delayedCar];

            await env.tacticalExecutor.executeTick(game, p0, {
                budgetSplit: [0, 0, 0, 1, 0],
                attackDesire: new Array(game.galaxy.stars.length).fill(0),
            });

            // 1. Paired high-traffic star s1 got the Warp Gate; isolated s4 did NOT.
            expect(s1.warpGate).toBeTrue();
            expect(s4.warpGate).toBeFalse();

            // 2. In-flight carrier kept locked hop[0] (s3 -> s0) and rerouted tail directly s0 -> s1 (dropAll)!
            expect(inFlightCar.waypoints.length).toBe(2);
            expect(inFlightCar.waypoints[0].destination.toString()).toBe(
                s0._id.toString(),
            );
            expect(inFlightCar.waypoints[1].source.toString()).toBe(
                s0._id.toString(),
            );
            expect(inFlightCar.waypoints[1].destination.toString()).toBe(
                s1._id.toString(),
            );
            expect(inFlightCar.waypoints[1].action).toBe("dropAll");

            // 3. Delayed ToT carrier was left untouched.
            expect(delayedCar.waypoints.length).toBe(3);
            expect(delayedCar.waypoints[1].destination.toString()).toBe(
                s2._id.toString(),
            );
        });
    });

    describe("AdmiralEnv cross-play, fog, recording & lookahead", () => {
        it("plays a deterministic mixed Admiral + Commander + AI game under fog and saves SBD1", async () => {
            const tmpFile = path.join(
                os.tmpdir(),
                `admiral_test_${Date.now()}.sbd`,
            );
            const runOnce = async (recordPath?: string) => {
                const env = new AdmiralEnv();
                let res = await env.reset({
                    admiralSeats: [0, 1],
                    commanderSeats: [2, 3],
                    seed: 42,
                    maxTicks: 48,
                    cycleOnly: true,
                    fog: true,
                    record: recordPath,
                });

                while (!res.done) {
                    for (const seat of res.seats) {
                        if (seat < 2) {
                            const obs = env.observeAdmiral(seat, {
                                truthFrame: true,
                            });
                            expect(obs.frame.owner.length).toBe(200);
                            expect(
                                obs.ownFacts.upgradeCosts.industry.length,
                            ).toBe(200);
                            expect(obs.truthFrame).toBeDefined();
                            expect(obs.truthFacts?.playerCredits.length).toBe(
                                10,
                            );
                            await env.actAdmiral(seat, {
                                budgetSplit: [0.25, 0.35, 0.25, 0.05, 0.1],
                            });
                        } else {
                            const obs = await env.observe(seat, {
                                frame: true,
                            });
                            expect(obs.frame).toBeDefined();
                            await env.act(seat, { explore: true });
                        }
                    }
                    res = await env.step();
                }
                return {
                    fp: fingerprint(env.engine, env.game),
                    standings: env.standings(),
                };
            };

            try {
                const a = await runOnce(tmpFile);
                const b = await runOnce();
                expect(a.fp).toEqual(b.fp);
                expect(a.standings.tick).toBe(48);

                expect(fs.existsSync(tmpFile)).toBeTrue();
                const raw = zlib.gunzipSync(fs.readFileSync(tmpFile));
                expect(raw.subarray(0, 4).toString("ascii")).toBe("SBD1");
            } finally {
                if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
            }
        });

        it("runs fog-legal lookaheadAdmiral without mutating game state or leaking hidden enemy ships", async () => {
            const env = new AdmiralEnv();
            await env.reset({
                admiralSeats: [0],
                commanderSeats: [1],
                seed: 77,
                maxTicks: 100,
                fog: true,
            });
            await env.actAdmiral(0, {});
            await env.act(1, {});
            await env.step(); // Advance to tick 1

            const obs = env.observeAdmiral(0);
            const S = obs.frame.owner.length;
            const world: BeliefWorld = {
                ships: new Array(S).fill(15),
                economy: new Array(S).fill(3),
                industry: new Array(S).fill(3),
                science: new Array(S).fill(1),
            };
            const plans = [
                { budgetSplit: [0.5, 0.2, 0.2, 0.0, 0.1], attackMargin: 1.2 },
                { budgetSplit: [0.1, 0.6, 0.2, 0.0, 0.1], attackMargin: 1.8 },
            ];

            const beforeFp = fingerprint(env.engine, env.game);
            const resA = await env.lookaheadAdmiral(0, plans, [world], 10, 123);
            expect(fingerprint(env.engine, env.game)).toEqual(beforeFp);
            expect(resA.length).toBe(2);
            expect(resA[0].length).toBe(1);

            // Mutate a strictly HIDDEN enemy star's true ships in env.game from its
            // real garrison to 99,999 and verify lookaheadAdmiral produces the EXACT
            // same rollout (proving zero fog leakage).
            const hiddenEnemyIdx = obs.frame.visibility.findIndex(
                (v: number, i: number) =>
                    v === HIDDEN && obs.frame.owner[i] > 0,
            );
            expect(hiddenEnemyIdx).toBeGreaterThanOrEqual(0);
            const hiddenStar = env.game.galaxy.stars[hiddenEnemyIdx];
            const savedShips = hiddenStar.ships;
            const savedActual = hiddenStar.shipsActual;
            try {
                hiddenStar.ships = 99_999;
                hiddenStar.shipsActual = 99_999;
                const resB = await env.lookaheadAdmiral(
                    0,
                    plans,
                    [world],
                    10,
                    123,
                );
                expect(resB).toEqual(resA);
            } finally {
                hiddenStar.ships = savedShips;
                hiddenStar.shipsActual = savedActual;
            }
        });
    });
});
