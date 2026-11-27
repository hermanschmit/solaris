import "../../env";
import fs from "fs";
import os from "os";
import path from "path";
import { execFileSync } from "child_process";
import mongoose from "mongoose";
import { SeatMemory } from "../../fog";
import { HIDDEN, VISIBLE, VISIBLE_NO_SHIPS } from "../../beliefData";
import {
    LiveAlert,
    LiveStateStore,
    PythonAdmiralBridge,
    buildDiplomaticPolicy,
    buildPeaceCancellationActions,
    computeResearchMasks,
    coordinateResearch,
    createEmptyHumanState,
    createShadowEngine,
    deserializeSeatMemory,
    detectReceivedTechnologies,
    estimateScheduledActionsReserve,
    hydrateShadowFromGalaxy,
    parseNotesDirectives,
    recordAiCommittedCarriers,
    recordShadowTickActions,
    serializeSeatMemory,
    updateCarrierPins,
} from "./index";

describe("Live Diplomatic Admiral stack", () => {
    describe("Notes # === AI === parser", () => {
        it("parses valid directives, ignores surrounding personal notes, and falls back safely on typos", () => {
            const raw = [
                "Personal reminder: Beta promised Weapons 3 next cycle!",
                "",
                "# === AI ===",
                "mode: autonomous        # active",
                "cash_reserve: 175",
                "attack: [RogueNine, 'Player Two']",
                "expect_tech: [weapons, scanning]",
                "neutral_border: defend",
                "# === END AI ===",
                "",
                "Other private notes below...",
            ].join("\n");

            const parsed = parseNotesDirectives(raw, null);
            expect(parsed.ok).toBeTrue();
            expect(parsed.hasBlock).toBeTrue();
            expect(parsed.directives.mode).toBe("autonomous");
            expect(parsed.directives.cashReserve).toBe(175);
            expect(parsed.directives.attack).toEqual([
                "RogueNine",
                "Player Two",
            ]);
            expect(parsed.directives.expectTech).toEqual([
                "weapons",
                "scanning",
            ]);
            expect(parsed.directives.neutralBorder).toBe("defend");

            // Typo inside block preserves `parsed.directives` and surfaces warning.
            const broken = [
                "# === AI ===",
                "mode: super_aggressive_invalid",
                "cash_reserve: -50",
                "# === END AI ===",
            ].join("\n");
            const fallback = parseNotesDirectives(broken, parsed.directives);
            expect(fallback.ok).toBeFalse();
            expect(fallback.fallbackUsed).toBeTrue();
            expect(fallback.warnings.length).toBeGreaterThanOrEqual(2);
            expect(fallback.directives.cashReserve).toBe(175);
            expect(fallback.directives.attack).toEqual([
                "RogueNine",
                "Player Two",
            ]);

            // Optional `# === END AI ===`: omitting the closing marker parses cleanly through EOF,
            // and `specialist_reserve` is supported.
            const noEnd = [
                "Personal reminder above",
                "# === AI ===",
                "mode: autonomous",
                "cash_reserve: 120",
                "specialist_reserve: 4",
            ].join("\n");
            const parsedNoEnd = parseNotesDirectives(noEnd);
            expect(parsedNoEnd.ok).toBeTrue();
            expect(parsedNoEnd.hasBlock).toBeTrue();
            expect(parsedNoEnd.warnings.length).toBe(0);
            expect(parsedNoEnd.directives.mode).toBe("autonomous");
            expect(parsedNoEnd.directives.cashReserve).toBe(120);
            expect(parsedNoEnd.directives.specialistReserve).toBe(4);
        });
    });

    describe("3-Tier Diplomatic Policy & Safety Invariants", () => {
        it("resolves ALLIED / PROTECTED / HOSTILE, protects pending offers, allows covert neutral attack, and hard-blocks allies in attack:", () => {
            const alerts: LiveAlert[] = [];
            const players = [
                { _id: "p0", alias: "Alpha" },
                { _id: "p1", alias: "AllyOne" },
                { _id: "p2", alias: "PendingBuddy" },
                { _id: "p3", alias: "QuietNeutral" },
                { _id: "p4", alias: "CovertTarget" },
                { _id: "p5", alias: "DeclaredFoe" },
            ];
            const statuses: any[] = [
                {
                    playerIdFrom: "p0",
                    playerIdTo: "p1",
                    playerFromAlias: "Alpha",
                    playerToAlias: "AllyOne",
                    statusFrom: "allies",
                    statusTo: "allies",
                    actualStatus: "allies",
                },
                {
                    playerIdFrom: "p0",
                    playerIdTo: "p2",
                    playerFromAlias: "Alpha",
                    playerToAlias: "PendingBuddy",
                    statusFrom: "neutral",
                    statusTo: "allies",
                    actualStatus: "neutral",
                },
                {
                    playerIdFrom: "p0",
                    playerIdTo: "p3",
                    playerFromAlias: "Alpha",
                    playerToAlias: "QuietNeutral",
                    statusFrom: "neutral",
                    statusTo: "neutral",
                    actualStatus: "neutral",
                },
                {
                    playerIdFrom: "p0",
                    playerIdTo: "p4",
                    playerFromAlias: "Alpha",
                    playerToAlias: "CovertTarget",
                    statusFrom: "neutral",
                    statusTo: "neutral",
                    actualStatus: "neutral",
                },
                {
                    playerIdFrom: "p0",
                    playerIdTo: "p5",
                    playerFromAlias: "Alpha",
                    playerToAlias: "DeclaredFoe",
                    statusFrom: "enemies",
                    statusTo: "neutral",
                    actualStatus: "enemies",
                },
            ];

            const policy = buildDiplomaticPolicy({
                myPlayerId: "p0",
                players,
                diplomaticStatuses: statuses,
                // Deliberately include formal ally `AllyOne` and covert neutral `CovertTarget`!
                attackList: ["AllyOne", "CovertTarget"],
                tick: 12,
                onAlert: (a) => alerts.push(a),
            });

            expect(policy.getTier("p1")).toBe("ALLIED");
            expect(policy.isHostile("p1")).toBeFalse();
            const allyAlert = alerts.find(
                (a) => a.code === "ALLY_IN_ATTACK_IGNORED",
            );
            expect(allyAlert).toBeDefined();
            expect(allyAlert!.message).toContain("'AllyOne'");
            expect(allyAlert!.message).not.toContain("(p1)");

            // Pending alliance offer (`statusTo === "allies"`, `actualStatus === "neutral"`)
            expect(policy.getTier("p2")).toBe("PROTECTED");
            expect(policy.isHostile("p2")).toBeFalse();

            // Default neutral -> PROTECTED
            expect(policy.getTier("p3")).toBe("PROTECTED");
            expect(policy.isHostile("p3")).toBeFalse();

            // Covert neutral override -> HOSTILE without calling declareEnemy
            expect(policy.getTier("p4")).toBe("HOSTILE");
            expect(policy.isHostile("p4")).toBeTrue();
            expect(policy.entriesByPlayerId.get("p4")?.covertAttack).toBeTrue();

            // Declared enemy -> HOSTILE
            expect(policy.getTier("p5")).toBe("HOSTILE");
            expect(policy.isHostile("p5")).toBeTrue();
        });

        it("cancels orbiting delayed attacks and truncates/alerts on unrecallable mid-flight carriers upon peace (Invariant 2)", () => {
            const alerts: LiveAlert[] = [];
            const players = [
                { _id: "p0", alias: "Alpha" },
                { _id: "p1", alias: "Beta" },
            ];
            const policy = buildDiplomaticPolicy({
                myPlayerId: "p0",
                players,
                diplomaticStatuses: [
                    {
                        playerIdFrom: "p0",
                        playerIdTo: "p1",
                        playerFromAlias: "Alpha",
                        playerToAlias: "Beta",
                        statusFrom: "neutral",
                        statusTo: "neutral",
                        actualStatus: "neutral",
                    },
                ],
                attackList: [],
                tick: 15,
            });

            const game = {
                galaxy: {
                    stars: [
                        { _id: "s0", name: "Home", ownedByPlayerId: "p0" },
                        { _id: "s1", name: "BetaCap", ownedByPlayerId: "p1" },
                        { _id: "s2", name: "BetaCore", ownedByPlayerId: "p1" },
                    ],
                    carriers: [
                        // 1. Orbiting ToT carrier with `delayTicks: 4` -> must cancel waypoints `[]`!
                        {
                            _id: "c_orbit",
                            name: "Alpha I",
                            ownedByPlayerId: "p0",
                            orbiting: "s0",
                            ships: 45,
                            isGift: false,
                            waypointsLooped: false,
                            waypoints: [
                                {
                                    source: "s0",
                                    destination: "s1",
                                    action: "nothing",
                                    actionShips: 0,
                                    delayTicks: 4,
                                },
                            ],
                        },
                        // 2. Mid-flight carrier heading `s0 -> s1 -> s2` -> must truncate `s2` and fire URGENT alert!
                        {
                            _id: "c_flight",
                            name: "Alpha II",
                            ownedByPlayerId: "p0",
                            orbiting: null,
                            ships: 60,
                            isGift: false,
                            waypointsLooped: false,
                            waypoints: [
                                {
                                    source: "s0",
                                    destination: "s1",
                                    action: "nothing",
                                    actionShips: 0,
                                    delayTicks: 0,
                                    ticksEta: 3,
                                },
                                {
                                    source: "s1",
                                    destination: "s2",
                                    action: "nothing",
                                    actionShips: 0,
                                    delayTicks: 0,
                                },
                            ],
                        },
                    ],
                },
            };

            const actions = buildPeaceCancellationActions({
                game,
                policy,
                tick: 15,
                onAlert: (a) => alerts.push(a),
                mutateShadow: true,
            });

            expect(actions.length).toBe(2);
            const orbitAct = actions.find(
                (a) => a.type === "SaveWaypoints" && a.carrierId === "c_orbit",
            ) as any;
            expect(orbitAct).toBeDefined();
            expect(orbitAct.carrierName).toBe("Alpha I");
            expect(orbitAct.waypoints).toEqual([]);
            expect(orbitAct.destinationNames).toEqual([]);

            const flightAct = actions.find(
                (a) => a.type === "SaveWaypoints" && a.carrierId === "c_flight",
            ) as any;
            expect(flightAct).toBeDefined();
            expect(flightAct.carrierName).toBe("Alpha II");
            expect(flightAct.waypoints.length).toBe(1);
            expect(flightAct.waypoints[0].destination).toBe("s1");
            expect(flightAct.destinationNames).toEqual(["BetaCap"]);

            const urgent = alerts.find((a) => a.code === "UNRECALLABLE_FLEET");
            expect(urgent).toBeDefined();
            expect(urgent!.severity).toBe("urgent");
        });

        it("enforces Invariants 1, 3 & 4 in TacticalExecutor (no neutral attacks/warp/transit, C2C reverse-lane block, gift/ally ignore, surprise-attack defense)", async () => {
            const { engine, executor } = createShadowEngine();
            engine.seed(88);
            const game = await engine.createGame(
                engine.loadSettings("standard"),
            );
            const p0 = game.galaxy.players[0]; // Us
            const p1 = game.galaxy.players[1]; // PROTECTED neutral
            const p2 = game.galaxy.players[2]; // ALLIED
            const p3 = game.galaxy.players[3]; // HOSTILE enemy

            p0.credits = 2000;
            p0.research.hyperspace.level = 8;
            p0.research.weapons.level = 3;
            game.galaxy.carriers = [];

            const stars = game.galaxy.stars;
            for (let idx = 0; idx < stars.length; idx++) {
                const s = stars[idx];
                s.ownedByPlayerId = null;
                s.ships = 0;
                s.shipsActual = 0;
                s.warpGate = false;
                s.wormHoleToStarId = null;
                s.isAsteroidField = false;
                // Isolate stars[8..] far out of jump range so only stars[0..7] participate!
                if (idx >= 8) {
                    s.location = { x: 50_000 + idx * 500, y: 50_000 };
                }
            }

            const speed = game.settings.specialGalaxy.carrierSpeed;
            // s0: our border star at (0, 0) with warpGate = true and 40 ships (defending against SneakNeutral)
            stars[0].ownedByPlayerId = p0._id;
            stars[0].location = { x: 0, y: 0 };
            stars[0].ships = 40;
            stars[0].shipsActual = 40;
            stars[0].warpGate = true;

            // s1: PROTECTED neutral star right next door at (speed * 2, 0) with 5 ships & warpGate!
            stars[1].ownedByPlayerId = p1._id;
            stars[1].location = { x: speed * 2, y: 0 };
            stars[1].ships = 5;
            stars[1].shipsActual = 5;
            stars[1].warpGate = true;

            // s2: Unclaimed star at (0, speed * 3), contested by p1's visible carrier!
            stars[2].ownedByPlayerId = null;
            stars[2].location = { x: 0, y: speed * 3 };

            // s3: Unclaimed star at (-speed * 3, 0), where p1 has an in-flight carrier flying `s3 -> s0`
            // (surprise attack on s0 AND active C2C lane `s3 -> s0`, blocking our `s0 -> s3` claim!).
            stars[3].ownedByPlayerId = null;
            stars[3].location = { x: -speed * 3, y: 0 };

            // s4: HOSTILE enemy star at (20_000, 0) with 10 ships -> ONLY valid invasion target!
            stars[4].ownedByPlayerId = p3._id;
            stars[4].location = { x: 20_000, y: 0 };
            stars[4].ships = 10;
            stars[4].shipsActual = 10;

            // s5: Isolated own enclave at (10_000, 10_000) (+ safe neighbor s6) receiving a 500-ship
            // ALLIED carrier & a 500-ship GIFT carrier 2 ticks away -> must NOT evacuate!
            stars[5].ownedByPlayerId = p0._id;
            stars[5].location = { x: 10_000, y: 10_000 };
            stars[5].ships = 12;
            stars[5].shipsActual = 12;
            stars[6].ownedByPlayerId = p0._id;
            stars[6].location = { x: 10_000 - speed * 2, y: 10_000 };
            stars[6].ships = 5;
            stars[6].shipsActual = 5;

            // s7: Un-attacked own shipyard at (20_000, speed * 2) with 120 ships in range of hostile s4
            // (plus PROTECTED neutral neighbor stars[8] right next to s7 to verify s7 also skips neutrals)!
            stars[7].ownedByPlayerId = p0._id;
            stars[7].location = { x: 20_000, y: speed * 2 };
            stars[7].ships = 120;
            stars[7].shipsActual = 120;

            // Visible neutral carrier contesting unclaimed s2:
            game.galaxy.carriers.push({
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p1._id,
                name: "NeutralClaimer",
                orbiting: null,
                isGift: false,
                ships: 5,
                specialistId: null,
                location: { x: 0, y: speed * 2 },
                locationNext: { x: 0, y: speed * 2.5 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: stars[1]._id,
                        destination: stars[2]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            } as any);

            // Visible neutral carrier flying `s3 -> s0` (surprise attack on s0 + reverse lane block on `s0 -> s3`):
            game.galaxy.carriers.push({
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p1._id,
                name: "SneakNeutral",
                orbiting: null,
                isGift: false,
                ships: 8,
                specialistId: null,
                location: { x: -speed * 1.5, y: 0 },
                locationNext: { x: -speed * 0.5, y: 0 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: stars[3]._id,
                        destination: stars[0]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            } as any);

            // Massive 500-ship GIFT carrier heading to s5 (must NEVER evacuate s5!):
            game.galaxy.carriers.push({
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p1._id,
                name: "FriendlyGift",
                orbiting: null,
                isGift: true,
                ships: 500,
                specialistId: null,
                location: { x: 10_000 - speed * 1.5, y: 10_000 },
                locationNext: { x: 10_000 - speed * 0.5, y: 10_000 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: stars[1]._id,
                        destination: stars[5]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            } as any);

            // Massive 500-ship ALLIED carrier heading to s5 (must NEVER evacuate s5!):
            game.galaxy.carriers.push({
                _id: new mongoose.Types.ObjectId(),
                ownedByPlayerId: p2._id,
                name: "AlliedTransit",
                orbiting: null,
                isGift: false,
                ships: 500,
                specialistId: null,
                location: { x: 10_000 - speed * 1.5, y: 10_000 },
                locationNext: { x: 10_000 - speed * 0.5, y: 10_000 },
                waypoints: [
                    {
                        _id: new mongoose.Types.ObjectId(),
                        source: stars[1]._id,
                        destination: stars[5]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                waypointsLooped: false,
            } as any);

            const policy = buildDiplomaticPolicy({
                myPlayerId: p0._id.toString(),
                players: game.galaxy.players,
                diplomaticStatuses: [
                    {
                        playerIdFrom: p0._id.toString(),
                        playerIdTo: p1._id.toString(),
                        playerFromAlias: p0.alias,
                        playerToAlias: p1.alias,
                        statusFrom: "neutral",
                        statusTo: "neutral",
                        actualStatus: "neutral",
                    },
                    {
                        playerIdFrom: p0._id.toString(),
                        playerIdTo: p2._id.toString(),
                        playerFromAlias: p0.alias,
                        playerToAlias: p2.alias,
                        statusFrom: "allies",
                        statusTo: "allies",
                        actualStatus: "allies",
                    },
                    {
                        playerIdFrom: p0._id.toString(),
                        playerIdTo: p3._id.toString(),
                        playerFromAlias: p0.alias,
                        playerToAlias: p3.alias,
                        statusFrom: "enemies",
                        statusTo: "enemies",
                        actualStatus: "enemies",
                    },
                ],
                tick: 2,
            });

            const execAlerts: any[] = [];
            await executor.executeTick(
                game,
                p0,
                {
                    attackDesire: new Array(stars.length).fill(5.0),
                    attackMargin: 1.4,
                    autoCarrierSpecialist: false,
                },
                undefined,
                null,
                {
                    diplomacy: policy.toExecutorHooks(),
                    onAlert: (a) => execAlerts.push(a),
                },
            );

            // 1. Surprise attack alert fired for `SneakNeutral` with alias (not raw ObjectId), yet ZERO counter-attacks on p1 (`stars[1]`)!
            const surpriseAlert = execAlerts.find(
                (a) => a.kind === "surprise_attack",
            );
            expect(surpriseAlert).toBeDefined();
            expect(surpriseAlert?.attackerAlias).toBe(p1.alias);
            expect(surpriseAlert?.message).toContain(`'${p1.alias}'`);
            expect(surpriseAlert?.message).not.toContain(p1._id.toString());

            const myCarriers = game.galaxy.carriers.filter(
                (c) => c.ownedByPlayerId?.toString() === p0._id.toString(),
            );
            const allDestinations = myCarriers.flatMap((c) =>
                c.waypoints.map((w) => w.destination.toString()),
            );

            // Never attacked PROTECTED neutral star s1:
            expect(allDestinations).not.toContain(stars[1]._id.toString());
            // Never contested neutral claim s2:
            expect(allDestinations).not.toContain(stars[2]._id.toString());
            // Never launched head-on into neutral C2C lane `s0 -> s3`:
            expect(allDestinations).not.toContain(stars[3]._id.toString());
            // DID launch invasion against HOSTILE enemy star s4:
            expect(allDestinations).toContain(stars[4]._id.toString());
            // Never evacuated s5 despite 500-ship gift + 500-ship ally inbound 2 ticks out:
            const evacFromS5 = myCarriers.find(
                (c) =>
                    c.waypoints?.[0]?.source.toString() ===
                        stars[5]._id.toString() &&
                    c.waypoints[0].action === "dropAll",
            );
            expect(evacFromS5).toBeUndefined();
            expect(stars[5].ships).toBe(12);
        });
    });

    describe("ResearchCoordinator, Cash Reserve, Shared-Seat Pins & CommandRecorder Coalescing", () => {
        it("coordinates received tech, manual GUI pins, ally tech gaps, and >=50% sunk-cost guard", async () => {
            const { engine } = createShadowEngine();
            engine.seed(41);
            const game = await engine.createGame(
                engine.loadSettings("standard"),
            );
            const me = game.galaxy.players[0];
            const ally = game.galaxy.players[1];
            (ally as any).isInScanningRange = true;
            ally.research.manufacturing.level = 4; // Ally already ahead on manufacturing!

            const techs = Object.keys(
                game.settings.technology.startingTechnologyLevel,
            ).filter((k) => /^[^_$]/.test(k));
            const human = createEmptyHumanState();
            for (const t of techs) {
                human.lastKnownTechLevels[t] = 1;
            }

            const diplomacy = buildDiplomaticPolicy({
                myPlayerId: me._id.toString(),
                players: game.galaxy.players,
                diplomaticStatuses: [
                    {
                        playerIdFrom: me._id.toString(),
                        playerIdTo: ally._id.toString(),
                        playerFromAlias: me.alias,
                        playerToAlias: ally.alias,
                        statusFrom: "allies",
                        statusTo: "allies",
                        actualStatus: "allies",
                    },
                ],
                tick: 5,
            });

            // Case A: Mid-cycle received technology (`weapons` jumped 1 -> 2, progress wiped to 0)!
            me.researchingNow = "weapons";
            me.researchQueue = ["random"];
            me.research.weapons.level = 2;
            me.research.weapons.progress = 0;

            const received = detectReceivedTechnologies({
                me,
                technologies: techs,
                human,
            });
            expect(received.length).toBe(1);
            expect(received[0].technology).toBe("weapons");

            const maskPlan = computeResearchMasks({
                engine,
                game,
                me,
                technologies: techs,
                diplomacy,
                expectTech: ["scanning"],
            });
            // Ally-held `manufacturing` and expected `scanning` are masked out!
            expect(
                maskPlan.allowedMask[1 + techs.indexOf("manufacturing")],
            ).toBeFalse();
            expect(
                maskPlan.allowedMask[1 + techs.indexOf("scanning")],
            ).toBeFalse();

            const alerts: LiveAlert[] = [];
            // Give highest logits to `manufacturing` (should be penalized as ally-held) and `hyperspace` (#1 eligible!)
            const logits = new Array(1 + techs.length).fill(1.0);
            logits[1 + techs.indexOf("manufacturing")] = 8.0;
            logits[1 + techs.indexOf("hyperspace")] = 6.0;
            logits[1 + techs.indexOf("terraforming")] = 5.0;

            const coord1 = coordinateResearch({
                engine,
                game,
                me,
                technologies: techs,
                human,
                maskPlan,
                receivedTechs: received,
                resLogits: logits,
                expectTech: ["scanning"],
                isCycleStart: false,
                onAlert: (a) => alerts.push(a),
            });

            expect(coord1.desiredNow).toBe("hyperspace");
            expect(coord1.desiredNext).toBe("terraforming");
            expect(coord1.desiredQueue[0]).toBe("terraforming");
            expect(coord1.desiredQueue.includes("random")).toBeFalse();
            expect(
                coord1.actions.some((a) => a.type === "SetResearchQueue"),
            ).toBeTrue();
            expect(
                alerts.some((a) => a.code === "TRADE_OPPORTUNITY"),
            ).toBeTrue();

            human.lastAiResearch = {
                now: coord1.desiredNow,
                next: coord1.desiredNext,
                queue: coord1.desiredQueue,
                tick: 5,
            };

            // Case A2: `hyperspace` levels up naturally (or via Experimentation) and Solaris
            // shifts `researchQueue[0]` (`terraforming`) into `researchingNow` -> must NOT
            // trigger a false-positive `HUMAN_RESEARCH_PIN` (even across consecutive ticks)!
            me.research.hyperspace.level += 1;
            me.researchingNow = "terraforming";
            me.researchQueue = coord1.desiredQueue.slice(1) as any;
            const pinAlertsBefore = alerts.filter(
                (a) => a.code === "HUMAN_RESEARCH_PIN",
            ).length;
            const coordNat = coordinateResearch({
                engine,
                game,
                me,
                technologies: techs,
                human,
                maskPlan,
                receivedTechs: [],
                resLogits: logits,
                isCycleStart: false,
                onAlert: (a) => alerts.push(a),
            });
            expect(coordNat.humanPinned).toBeFalse();
            expect(
                alerts.filter((a) => a.code === "HUMAN_RESEARCH_PIN").length,
            ).toBe(pinAlertsBefore);
            // Follow-up tick without dispatching (e.g., advisor mode) must also stay unpinned:
            const coordNatNext = coordinateResearch({
                engine,
                game,
                me,
                technologies: techs,
                human,
                maskPlan,
                receivedTechs: [],
                resLogits: logits,
                isCycleStart: false,
                onAlert: (a) => alerts.push(a),
            });
            expect(coordNatNext.humanPinned).toBeFalse();
            expect(
                alerts.filter((a) => a.code === "HUMAN_RESEARCH_PIN").length,
            ).toBe(pinAlertsBefore);

            // Case B: Human manually changes `researchingNow` to `banking` in GUI!
            me.researchingNow = "banking";
            me.researchQueue = ["terraforming"];
            const coord2 = coordinateResearch({
                engine,
                game,
                me,
                technologies: techs,
                human,
                maskPlan,
                receivedTechs: [],
                resLogits: logits,
                isCycleStart: true,
                onAlert: (a) => alerts.push(a),
            });
            expect(coord2.humanPinned).toBeTrue();
            expect(coord2.desiredNow).toBe("banking");
            expect(
                alerts.filter((a) => a.code === "HUMAN_RESEARCH_PIN").length,
            ).toBe(pinAlertsBefore + 1);
            expect(
                coord2.actions.some((a) => a.type === "SetResearchNow"),
            ).toBeFalse();
        });

        it("protects human-created carriers across grace ticks, respects ignoreBulkUpgrade & cashReserve, and coalesces CommandRecorder actions", async () => {
            const { engine, executor } = createShadowEngine();
            engine.seed(52);
            const game = await engine.createGame(
                engine.loadSettings("standard"),
            );
            game.state.tick = 1; // Cycle start spending tick!
            const me = game.galaxy.players[0];
            const foe = game.galaxy.players[1];
            me.credits = 600;
            me.research.hyperspace.level = 6;
            game.galaxy.carriers = [];

            const ownStars = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === me._id.toString(),
            );
            const starA = ownStars[0];
            const starB = ownStars[1];
            for (const s of ownStars) {
                s.ships = 5;
                s.shipsActual = 5;
            }
            starA.ships = 40;
            starA.shipsActual = 40;
            // Mark starA as Ignore Bulk Upgrade for science & industry in GUI!
            starA.ignoreBulkUpgrade = {
                economy: true,
                industry: true,
                science: true,
            };

            // Place a hostile target star in range of starA so Admiral launches an attack from starA:
            const foeStar = game.galaxy.stars.find(
                (s) => s.ownedByPlayerId?.toString() === foe._id.toString(),
            )!;
            const speed = game.settings.specialGalaxy.carrierSpeed;
            starA.location = { x: 0, y: 0 };
            foeStar.location = { x: speed * 2, y: 0 };
            foeStar.ships = 5;
            foeStar.shipsActual = 5;
            foeStar.infrastructure = { economy: 1, industry: 0, science: 0 };

            // Human built a carrier at starB with 30 ships in the GUI (`gui_created`)!
            const humanCarrierId = new mongoose.Types.ObjectId();
            game.galaxy.carriers.push({
                _id: humanCarrierId,
                ownedByPlayerId: me._id,
                name: "Gift Staging",
                orbiting: starB._id,
                isGift: false,
                ships: 30,
                specialistId: null,
                location: { ...starB.location },
                locationNext: { ...starB.location },
                waypoints: [],
                waypointsLooped: false,
            } as any);

            // Also an existing AI idle carrier at starA with 20 ships (will be reused/reloaded):
            const aiIdleId = new mongoose.Types.ObjectId();
            game.galaxy.carriers.push({
                _id: aiIdleId,
                ownedByPlayerId: me._id,
                name: "AI Ferry",
                orbiting: starA._id,
                isGift: false,
                ships: 20,
                specialistId: null,
                location: { ...starA.location },
                locationNext: { ...starA.location },
                waypoints: [],
                waypointsLooped: false,
            } as any);

            const human = createEmptyHumanState();
            // Record only `aiIdleId` as previously AI-committed at tick 0 -> `humanCarrierId` is detected as GUI-created!
            recordAiCommittedCarriers({
                postShadowCarriers: [game.galaxy.carriers[1]],
                myPlayerId: me._id.toString(),
                pinnedCarrierIds: new Set(),
                remappedCarrierIds: new Map(),
                human,
                tick: 0,
            });

            const pinned = updateCarrierPins({
                ownCarriers: game.galaxy.carriers,
                human,
                tick: 1,
                lastProcessedTick: 0,
                graceTicks: 2,
            });
            expect(pinned.has(humanCarrierId.toString())).toBeTrue();
            expect(pinned.has(aiIdleId.toString())).toBeFalse();

            // Add a scheduled bulk action costing 50cr + cashReserve 150cr = 200cr protected!
            (me as any).scheduledActions = [
                {
                    _id: new mongoose.Types.ObjectId(),
                    infrastructureType: "economy",
                    buyType: "totalCredits",
                    amount: 50,
                    repeat: false,
                    tick: 19,
                    active: true,
                },
            ];
            const schedReserve = estimateScheduledActionsReserve(
                engine,
                game,
                me,
            );
            expect(schedReserve).toBe(50);

            const diplomacy = buildDiplomaticPolicy({
                myPlayerId: me._id.toString(),
                players: game.galaxy.players,
                diplomaticStatuses: [
                    {
                        playerIdFrom: me._id.toString(),
                        playerIdTo: foe._id.toString(),
                        playerFromAlias: me.alias,
                        playerToAlias: foe.alias,
                        statusFrom: "enemies",
                        statusTo: "enemies",
                        actualStatus: "enemies",
                    },
                ],
                tick: 1,
            });

            const mem = new SeatMemory(game.galaxy.stars.length);
            mem.update(engine, game, me);
            const starAPreSci = starA.infrastructure.science;
            const starAPreInd = starA.infrastructure.industry;

            const foeIdx = game.galaxy.stars.indexOf(foeStar);
            const desire = new Array(game.galaxy.stars.length).fill(0);
            desire[foeIdx] = 10.0;

            const plan = await recordShadowTickActions({
                engine,
                executor,
                shadowGame: game,
                mySeatIndex: 0,
                seatMemory: mem,
                view: mem.view!,
                decision: {
                    budgetSplit: [0.4, 0.4, 0.1, 0.05, 0.05],
                    attackDesire: desire,
                    attackMargin: 1.35,
                    autoCarrierSpecialist: false,
                },
                researchActions: [
                    { type: "SetResearchNow", preference: "weapons" },
                ],
                diplomacy,
                pinnedCarrierIds: pinned,
                cashReserveCredits: 150,
                scheduledActionCredits: schedReserve,
            });

            // 1. Protected cash reserve (150 + 50 = 200cr) strictly preserved!
            expect(me.credits).toBeGreaterThanOrEqual(200);

            // 2. `starA` had `ignoreBulkUpgrade = true` -> zero upgrades on starA!
            expect(starA.infrastructure.science).toBe(starAPreSci);
            expect(starA.infrastructure.industry).toBe(starAPreInd);
            expect(
                plan.actions.some(
                    (a) =>
                        a.type === "UpgradeStar" &&
                        a.starId === starA._id.toString(),
                ),
            ).toBeFalse();

            // 3. Human-created carrier at starB STILL has all 30 ships untouched!
            const humanCarAfter = game.galaxy.carriers.find(
                (c) => c._id.toString() === humanCarrierId.toString(),
            )!;
            expect(humanCarAfter.ships).toBe(30);
            expect(
                plan.actions.some(
                    (a) =>
                        (a.type === "TransferShips" ||
                            a.type === "SaveWaypoints") &&
                        a.carrierId === humanCarrierId.toString(),
                ),
            ).toBeFalse();

            // 4. Verify dependency ordering: SetResearchNow comes before UpgradeStar, which comes before SaveWaypoints!
            const firstResIdx = plan.actions.findIndex(
                (a) => a.type === "SetResearchNow",
            );
            const firstUpgIdx = plan.actions.findIndex(
                (a) => a.type === "UpgradeStar",
            );
            const firstWpIdx = plan.actions.findIndex(
                (a) => a.type === "SaveWaypoints",
            );
            expect(firstResIdx).toBe(0);
            if (firstUpgIdx >= 0 && firstWpIdx >= 0) {
                expect(firstUpgIdx).toBeLessThan(firstWpIdx);
            }
        });

        it("guards scanning specialists (#8), enforces specialist_reserve with battle-swing dips (#7), and attaches human-readable action names (#1, #5)", async () => {
            const { engine, executor } = createShadowEngine();
            engine.seed(58);
            const game = await engine.createGame(
                engine.loadSettings("standard"),
            );
            game.state.tick = 10;
            const me = game.galaxy.players[0];
            const foe = game.galaxy.players[1];
            const ly = game.constants?.distances?.lightYear || 50;
            me.credits = 500;
            me.creditsSpecialists = 3;
            me.research.scanning.level = 1;
            me.research.weapons.level = 1;
            foe.research.weapons.level = 1;
            game.galaxy.carriers = [];

            const ownStars = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === me._id.toString(),
            );
            const interiorStar = ownStars[0];
            const frontStar = ownStars[1];
            const foeStar = game.galaxy.stars.find(
                (s) => s.ownedByPlayerId?.toString() === foe._id.toString(),
            )!;
            const interiorIdx = game.galaxy.stars.indexOf(interiorStar);
            const foeIdx = game.galaxy.stars.indexOf(foeStar);

            for (const s of game.galaxy.stars) {
                s.specialistId = null;
                s.ships = 1;
                s.shipsActual = 1;
                if (s !== interiorStar && s !== frontStar && s !== foeStar) {
                    s.location = { x: -8000, y: -8000 };
                }
            }
            interiorStar.location = { x: 0, y: 0 };
            frontStar.location = { x: 10 * ly, y: 0 };
            foeStar.location = { x: 12.2 * ly, y: 0 };

            const viewAllVisible = {
                stars: new Uint8Array(game.galaxy.stars.length).fill(1),
                carriers: new Set<any>(),
                carrierShips: new Set<any>(),
            } as any;
            const dipPolicy = buildDiplomaticPolicy({
                myPlayerId: me._id.toString(),
                players: game.galaxy.players,
                diplomaticStatuses: [
                    {
                        playerIdFrom: me._id.toString(),
                        playerIdTo: foe._id.toString(),
                        statusFrom: "enemies",
                        statusTo: "enemies",
                        actualStatus: "enemies",
                    } as any,
                ],
                attackList: [],
                neutralBorderPosture: "defend",
                tick: 10,
            });
            const mem = new SeatMemory(game.galaxy.stars.length);
            mem.update(engine, game, me);

            // Case 1 (#8): Orbital Array (id=1, pure +scanning) on interiorStar reveals 0 hidden stars -> skipped!
            const planScanSkip = await recordShadowTickActions({
                engine,
                executor,
                shadowGame: game,
                mySeatIndex: 0,
                seatMemory: mem,
                view: viewAllVisible,
                decision: {
                    hire: [{ star: interiorIdx, specialist: 1 }],
                    attackDesire: new Array(game.galaxy.stars.length).fill(0),
                },
                researchActions: [],
                diplomacy: dipPolicy,
                pinnedCarrierIds: new Set(),
                cashReserveCredits: 100,
                scheduledActionCredits: 0,
                specialistReserveTokens: 0,
            });
            expect(
                planScanSkip.actions.some(
                    (a) => a.type === "HireStarSpecialist",
                ),
            ).toBeFalse();

            // Case 2 (#7): specialistReserveTokens=3 blocks routine Space Dock (id=2, cost 2),
            // YET allows emergency defensive Orbital Cannon (id=4, +1 defenderBonus, cost 2)
            // when an inbound foe carrier (28 ships vs 12 defending ships, weapons 1 vs 1:
            // loses at w2 vs w1 since 12*2=24 < 28, wins with Orbital Cannon at w3 vs w1 since 12*3=36 >= 28!).
            frontStar.ships = 12;
            frontStar.shipsActual = 12;
            const speed = game.settings.specialGalaxy.carrierSpeed;
            game.galaxy.carriers = [
                {
                    _id: new mongoose.Types.ObjectId(),
                    ownedByPlayerId: foe._id,
                    name: "FoeRaider",
                    orbiting: null,
                    isGift: false,
                    ships: 28,
                    specialistId: null,
                    specialistExpireTick: null,
                    specialist: null,
                    location: { x: 10 * ly + speed * 1.5, y: 0 },
                    locationNext: { x: 10 * ly + speed * 0.5, y: 0 },
                    waypoints: [
                        {
                            _id: new mongoose.Types.ObjectId(),
                            source: foeStar._id,
                            destination: frontStar._id,
                            action: "nothing",
                            actionShips: 0,
                            delayTicks: 0,
                        },
                    ],
                    waypointsLooped: false,
                } as any,
            ];
            viewAllVisible.stars[foeIdx] = 1;
            viewAllVisible.carriers.add(game.galaxy.carriers[0]);
            viewAllVisible.carrierShips.add(game.galaxy.carriers[0]);
            mem.update(engine, game, me);

            const planReserve = await recordShadowTickActions({
                engine,
                executor,
                shadowGame: game,
                mySeatIndex: 0,
                seatMemory: mem,
                view: viewAllVisible,
                decision: {
                    hire: [{ star: interiorIdx, specialist: 2 }],
                    attackDesire: new Array(game.galaxy.stars.length).fill(0),
                },
                researchActions: [],
                diplomacy: dipPolicy,
                pinnedCarrierIds: new Set(),
                cashReserveCredits: 100,
                scheduledActionCredits: 0,
                specialistReserveTokens: 3,
                allowEmergencySpecialistDip: true,
            });

            expect(
                planReserve.actions.some(
                    (a) =>
                        a.type === "HireStarSpecialist" &&
                        a.starId === interiorStar._id.toString(),
                ),
            ).toBeFalse();

            const defHire = planReserve.actions.find(
                (a) =>
                    a.type === "HireStarSpecialist" &&
                    a.starId === frontStar._id.toString(),
            ) as any;
            expect(defHire).toBeDefined();
            expect(defHire.specialistName).toBeTruthy();
            expect(defHire.starName).toBe(frontStar.name);
            expect(
                planReserve.alerts.some(
                    (a) => a.code === "EMERGENCY_SPECIALIST_DIP",
                ),
            ).toBeTrue();
        });
    });

    describe("SQLite/JSON Persistence, Nebula/Stealth Hydration & Python AdmiralNet RPC", () => {
        it("round-trips SQLite (.db) and JSON (.json) state stores, maps nebula ships:null to VISIBLE_NO_SHIPS, and runs Python diplomatic inference + checkpoint loading + lookahead", async () => {
            const tmpDir = fs.mkdtempSync(
                path.join(os.tmpdir(), "admiral_live_spec_"),
            );
            const bridge = new PythonAdmiralBridge();

            try {
                const { engine } = createShadowEngine();
                engine.seed(64);
                const game = await engine.createGame(
                    engine.loadSettings("standard"),
                );
                const p0 = game.galaxy.players[0];
                // Ensure p0 has a valid userId for `engine.viewAs`
                const uid = new mongoose.Types.ObjectId();
                engine.setExternallyControlled(game, p0._id, uid);

                // Force an opponent star inside scanning range to be a Nebula (`ships: null` in `/galaxy`)!
                const p0Home = game.galaxy.stars.find(
                    (s) => s._id.toString() === p0.homeStarId?.toString(),
                )!;
                const oppStar = game.galaxy.stars.find(
                    (s) =>
                        s.ownedByPlayerId &&
                        s.ownedByPlayerId.toString() !== p0._id.toString(),
                )!;
                oppStar.location = {
                    x: p0Home.location.x + 5,
                    y: p0Home.location.y + 5,
                };
                oppStar.isNebula = true;
                oppStar.ships = 42;
                oppStar.shipsActual = 42;

                const liveGalaxyJson = await engine.viewAs(game, uid);
                const oppIdx = liveGalaxyJson.galaxy.stars.findIndex(
                    (s: any) => s._id.toString() === oppStar._id.toString(),
                );
                expect(liveGalaxyJson.galaxy.stars[oppIdx].ships).toBeNull();

                const mem = new SeatMemory(liveGalaxyJson.galaxy.stars.length);
                const hyd = hydrateShadowFromGalaxy({
                    engine,
                    galaxyPayload: liveGalaxyJson,
                    seatMemory: mem,
                });

                // Nebula star in scan range MUST be `VISIBLE_NO_SHIPS` (not `VISIBLE` with 0 ships!),
                // and must NOT be misclassified as a dead star!
                expect(hyd.view.stars[oppIdx]).toBe(VISIBLE_NO_SHIPS);
                expect(
                    engine.container.starDataService.isDeadStar(
                        hyd.shadowGame.galaxy.stars[oppIdx],
                    ),
                ).toBeFalse();
                // And hidden stars outside scan range remain `HIDDEN`:
                expect(
                    Array.from(hyd.view.stars).some((v) => v === HIDDEN),
                ).toBeTrue();
                expect(
                    Array.from(hyd.view.stars).some((v) => v === VISIBLE),
                ).toBeTrue();

                // Save a real `AdmiralNet` checkpoint `.pt` file and load it via `PythonAdmiralBridge`!
                const ckptPath = path.join(tmpDir, "test_ema.pt");
                execFileSync("python3", [
                    "-c",
                    [
                        "import sys, torch",
                        `sys.path.insert(0, ${JSON.stringify(path.join(__dirname, "..", "python"))})`,
                        `sys.path.insert(0, ${JSON.stringify(path.join(__dirname, "..", "..", "python"))})`,
                        "from admiral_net import AdmiralNet",
                        "m = AdmiralNet(8, d=64, n_layers=2)",
                        `torch.save({'config': {'n_techs': 8, 'd': 64, 'n_layers': 2, 'n_heads': 4}, 'model': m.state_dict()}, ${JSON.stringify(ckptPath)})`,
                    ].join("; "),
                ]);

                const initInfo = await bridge.init({
                    header: hyd.staticHeader,
                    seat: hyd.mySeatIndex,
                    checkpoint: ckptPath,
                });
                expect(initInfo.stars).toBe(200);
                expect(initInfo.players).toBe(10);

                // All 9 opponents PROTECTED except player 3 (`HOSTILE`):
                const tiers = new Array(10).fill("PROTECTED") as any[];
                tiers[hyd.mySeatIndex] = "ALLIED";
                tiers[3] = "HOSTILE";

                const decOut = await bridge.decide({
                    obs: hyd.observation,
                    diplomaticTiers: tiers,
                    neutralBorderPosture: "defend",
                    bannedStarSpecialists: [16, 17],
                });

                expect(decOut.decision.court).toBeFalse();
                expect(decOut.decision.betray).toBeFalse();
                expect(decOut.resLogits.length).toBe(9);
                expect(decOut.trackerState.history.length).toBe(1);

                // Invariant 5: Every star owned by a PROTECTED player has `attackDesire === 0`!
                hyd.observation.frame.owner.forEach(
                    (ownerIdx: number, starIdx: number) => {
                        if (
                            ownerIdx >= 0 &&
                            ownerIdx !== hyd.mySeatIndex &&
                            ownerIdx !== 3
                        ) {
                            expect(
                                decOut.decision.attackDesire?.[starIdx],
                            ).toBe(0);
                        }
                    },
                );

                // Verify 1-cycle lookahead proposal + leaf scoring works with the checkpoint:
                const searchProp = await bridge.proposeSearch({
                    obs: hyd.observation,
                    diplomaticTiers: tiers,
                    neutralBorderPosture: "defend",
                    nPlans: 3,
                    nWorlds: 2,
                    seed: 7,
                });
                expect(searchProp.plans.length).toBe(3);
                expect(searchProp.worlds.length).toBe(2);

                const leafScores = await bridge.scoreLeaves({
                    leaves: [
                        {
                            frame: hyd.observation.frame,
                            ownFacts: hyd.observation.ownFacts,
                            players: hyd.observation.frame.stats.map(
                                (st: number[], idx: number) => ({
                                    stars: st[0],
                                    ships: st[1],
                                    defeated: Boolean(
                                        hyd.observation.frame.defeated[idx],
                                    ),
                                }),
                            ),
                            ticksToEnd: null,
                        },
                    ],
                    diplomaticTiers: tiers,
                    neutralBorderPosture: "defend",
                });
                expect(leafScores.scores.length).toBe(1);
                expect(Number.isFinite(leafScores.scores[0])).toBeTrue();

                // Verify BOTH SQLite (`.db`) and JSON (`.json`) persistence round-trip!
                for (const ext of ["db", "json"]) {
                    const storePath = path.join(tmpDir, `state_test.${ext}`);
                    const store = new LiveStateStore(storePath);
                    const loaded0 = store.load("g1", "p0");
                    loaded0.lastProcessedTick = 14;
                    loaded0.seatMemory = serializeSeatMemory(mem);
                    loaded0.beliefTracker = decOut.trackerState;
                    loaded0.human.lastAiResearch = {
                        now: "weapons",
                        next: "hyperspace",
                        tick: 14,
                    };
                    store.save(loaded0);
                    store.close();

                    const store2 = new LiveStateStore(storePath);
                    const restored = store2.load("g1", "p0");
                    expect(restored.lastProcessedTick).toBe(14);
                    expect(restored.beliefTracker?.history.length).toBe(1);
                    expect(restored.human.lastAiResearch.now).toBe("weapons");
                    const memRestored = deserializeSeatMemory(
                        restored.seatMemory,
                        200,
                    );
                    expect(Array.from(memRestored.seenTick)).toEqual(
                        Array.from(mem.seenTick),
                    );
                    store2.close();
                }
            } finally {
                bridge.close();
                fs.rmSync(tmpDir, { recursive: true, force: true });
            }
        }, 30_000);
    });

    describe("Scheduled Assaults (attack: [Target@tick])", () => {
        it("parses Target@tick in Notes, rejects invalid @tick syntax, and enforces Invariant 1 for allies", () => {
            const good = parseNotesDirectives(
                "# === AI ===\nattack: [EnemyNow, Beta@120]\n# === END AI ===",
            );
            expect(good.ok).toBeTrue();
            expect(good.directives.attack).toEqual(["EnemyNow", "Beta@120"]);

            const bad = parseNotesDirectives(
                "# === AI ===\nattack: [Beta@abc]\n# === END AI ===",
                good.directives,
            );
            expect(bad.ok).toBeFalse();
            expect(bad.fallbackUsed).toBeTrue();
            expect(bad.directives.attack).toEqual(["EnemyNow", "Beta@120"]);
            expect(
                bad.warnings.some((w) => w.includes("Invalid attack target")),
            ).toBeTrue();

            const alerts: LiveAlert[] = [];
            const policy = buildDiplomaticPolicy({
                myPlayerId: "p0",
                players: [
                    { _id: "p0", alias: "Alpha" },
                    { _id: "p1", alias: "AllyOne" },
                    { _id: "p2", alias: "Beta" },
                ],
                diplomaticStatuses: [
                    {
                        playerIdFrom: "p0",
                        playerIdTo: "p1",
                        playerFromAlias: "Alpha",
                        playerToAlias: "AllyOne",
                        statusFrom: "allies",
                        statusTo: "allies",
                        actualStatus: "allies",
                    },
                    {
                        playerIdFrom: "p0",
                        playerIdTo: "p2",
                        playerFromAlias: "Alpha",
                        playerToAlias: "Beta",
                        statusFrom: "neutral",
                        statusTo: "neutral",
                        actualStatus: "neutral",
                    },
                ],
                attackList: ["AllyOne@120", "Beta@120"],
                tick: 100,
                onAlert: (a) => alerts.push(a),
            });

            expect(policy.getTier("p1")).toBe("ALLIED");
            expect(policy.getAttackStartTick("p1")).toBeNull();
            expect(
                alerts.some((a) => a.code === "ALLY_IN_ATTACK_IGNORED"),
            ).toBeTrue();

            expect(policy.getTier("p2")).toBe("HOSTILE");
            expect(policy.getAttackStartTick("p2")).toBe(120);
        });

        it("stages interior ships to the border before attackStartTick without launching, then launches on time for landing >= attackStartTick", async () => {
            const { engine, executor } = createShadowEngine();
            engine.seed(77);
            const game = await engine.createGame(
                engine.loadSettings("standard"),
            );
            const p0 = game.galaxy.players[0];
            const p1 = game.galaxy.players[1];

            p0.credits = 500;
            p0.research.hyperspace.level = 1; // hsDist = 2.5 LY
            p0.research.weapons.level = 3;
            game.galaxy.carriers = [];

            const ly = game.constants.distances.lightYear;
            const stars = game.galaxy.stars;
            for (const s of stars) {
                s.ownedByPlayerId = null;
                s.ships = 0;
                s.shipsActual = 0;
                s.location = { x: 9000, y: 9000 };
                s.infrastructure = { economy: 0, industry: 0, science: 0 };
            }

            // Interior (0 LY) -> Border (2.0 LY) -> Beta (3.0 LY):
            // - Interior is 3.0 LY > hsDist (2.5 LY) from Beta, so it is a true interior source.
            // - Border -> Beta is 1.0 LY (10 ticks at standard speed).
            const interior = stars[0];
            const border = stars[1];
            const betaStar = stars[2];

            interior.ownedByPlayerId = p0._id;
            interior.ships = 80;
            interior.shipsActual = 80;
            interior.location = { x: 0, y: 0 };

            border.ownedByPlayerId = p0._id;
            border.ships = 20;
            border.shipsActual = 20;
            border.location = { x: 2.0 * ly, y: 0 };

            betaStar.ownedByPlayerId = p1._id;
            betaStar.ships = 10;
            betaStar.shipsActual = 10;
            betaStar.location = { x: 3.0 * ly, y: 0 };

            const policy = buildDiplomaticPolicy({
                myPlayerId: p0._id.toString(),
                players: game.galaxy.players.map((p) => ({
                    _id: p._id.toString(),
                    alias: p.alias,
                })),
                diplomaticStatuses: [
                    {
                        playerIdFrom: p0._id.toString(),
                        playerIdTo: p1._id.toString(),
                        playerFromAlias: p0.alias,
                        playerToAlias: p1.alias,
                        statusFrom: "neutral",
                        statusTo: "neutral",
                        actualStatus: "neutral",
                    },
                ],
                attackList: [`${p1.alias}@120`],
                tick: 100,
            });

            // Phase 1a (tick 100): Interior stages ships to Border; zero premature attacks.
            game.state.tick = 100;
            await executor.executeTick(
                game,
                p0,
                {
                    attackDesire: stars.map(() => 1.0),
                    attackMargin: 1.5,
                    budgetSplit: [0, 0, 0, 0, 1],
                },
                undefined,
                undefined,
                { diplomacy: policy.toExecutorHooks() },
            );

            const betaId = betaStar._id.toString();
            const borderId = border._id.toString();
            const prematureAttacks1 = game.galaxy.carriers.filter((c) =>
                (c.waypoints ?? []).some(
                    (w) => w.destination.toString() === betaId,
                ),
            );
            expect(prematureAttacks1.length).toBe(0);

            const stagingCarriers = game.galaxy.carriers.filter((c) =>
                (c.waypoints ?? []).some(
                    (w) => w.destination.toString() === borderId,
                ),
            );
            expect(stagingCarriers.length).toBeGreaterThanOrEqual(1);

            const borderDist =
                engine.container.distanceService.getDistanceBetweenLocations(
                    border.location,
                    betaStar.location,
                );
            const borderEta = Math.max(
                1,
                Math.ceil(
                    borderDist / game.settings.specialGalaxy.carrierSpeed,
                ),
            );

            // Phase 1b: 1 tick BEFORE launch window — even with 80 ships on Border,
            // launching now would arrive at tick 119 < 120, so it must stay deferred!
            border.ships = 80;
            border.shipsActual = 80;
            p0.credits = 500;
            game.state.tick = 120 - borderEta - 1;
            await executor.executeTick(
                game,
                p0,
                {
                    attackDesire: stars.map(() => 1.0),
                    attackMargin: 1.5,
                    budgetSplit: [0, 0, 0, 0, 1],
                },
                undefined,
                undefined,
                { diplomacy: policy.toExecutorHooks() },
            );

            const prematureAttacks2 = game.galaxy.carriers.filter((c) =>
                (c.waypoints ?? []).some(
                    (w) => w.destination.toString() === betaId,
                ),
            );
            expect(prematureAttacks2.length).toBe(0);

            // Phase 2: exact launch tick (120 - borderEta) -> arrives at tick 120!
            p0.credits = 500;
            game.state.tick = 120 - borderEta;
            await executor.executeTick(
                game,
                p0,
                {
                    attackDesire: stars.map(() => 1.0),
                    attackMargin: 1.5,
                    budgetSplit: [0, 0, 0, 0, 1],
                },
                undefined,
                undefined,
                { diplomacy: policy.toExecutorHooks() },
            );

            const launchedAttacks = game.galaxy.carriers.filter((c) =>
                (c.waypoints ?? []).some(
                    (w) => w.destination.toString() === betaId,
                ),
            );
            expect(launchedAttacks.length).toBeGreaterThanOrEqual(1);
        });
    });
});
