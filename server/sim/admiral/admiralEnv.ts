// Multi-protocol simulation environment supporting both:
//   - Admiral seats (driven by TacticalExecutor + AdmiralDecision), and
//   - Commander seats (driven by the existing AgentEnv + AIService executor),
// side by side in the same game under fog of war, plus fog-legal
// belief-determinized lookahead rollouts for cycle-boundary planning.
//
// Subclasses AgentEnv without modifying any upstream or sim file.

import { ValidationError } from "@solaris/common";
import { AgentEnv, AgentDecision, ResetOptions, StepResult } from "../agentEnv";
import { seatFrame, fullView, SeatMemory } from "../fog";
import { VISIBLE, HIDDEN } from "../beliefData";
import { Game } from "../../services/types/Game";
import { Player } from "../../services/types/Player";
import { Star, InfrastructureType } from "../../services/types/Star";
import { TacticalExecutor, AdmiralDecision } from "./tacticalExecutor";
import { installFastCombat } from "./fastDefense";

export interface AdmiralResetOptions extends Omit<ResetOptions, "agents"> {
    // Seats controlled by the new Admiral tactical executor.
    admiralSeats?: number[];
    // Seats controlled by the existing Commander (AIService) flow.
    commanderSeats?: number[];
    // Legacy alias for commanderSeats if passed by existing scripts.
    agents?: number[];
}

export interface BeliefWorld {
    // Full-galaxy arrays (length S) filled from a conservation-consistent
    // belief sample for hidden stars (visible stars keep their known values).
    ships: number[];
    economy?: number[];
    industry?: number[];
    science?: number[];
}

export interface AdmiralOwnFacts {
    credits: number;
    specialistBudget: number;
    carrierCost: number;
    researchingNow: number; // index into technologies
    researchingNext: number;
    researchProgress: number[]; // per technology
    researchRequired: number[]; // total points needed for next level, per technology
    affordableStarSpecialists: number[];
    affordableCarrierSpecialists: number[];
    // Marginal upgrade cost at each star (-1 if unowned or unavailable).
    upgradeCosts: {
        economy: number[];
        industry: number[];
        science: number[];
        warpGate: number[];
    };
    ownCarriers: {
        id: string;
        ships: number;
        orbiting: number;
        destination: number;
        delayTicks: number;
        specialist: number;
    }[];
    incomingAttacks: {
        starIndex: number;
        ticksUntil: number;
        ships: number;
    }[];
}

export interface AdmiralTruthFacts {
    playerCredits: number[];
    playerSpecTokens: number[];
    playerResearchingNow: number[];
    playerResearchProgress: number[][];
}

export interface AdmiralObservation {
    seat: number;
    tick: number;
    cycle: number;
    ticksToProduction: number;
    ticksToEnd: number | null;
    defeated: boolean;
    frame: any;
    ownFacts: AdmiralOwnFacts;
    truthFrame?: any;
    truthFacts?: AdmiralTruthFacts;
}

export class AdmiralEnv extends AgentEnv {
    tacticalExecutor: TacticalExecutor;
    admiralSeatSet = new Set<number>();
    admiralDecisions = new Map<number, AdmiralDecision>();
    private basePlayAI!: (eventService: any, game: Game) => Promise<void>;

    constructor() {
        // Capture raw GameTickService._playAI before super() wraps it, so we can
        // dispatch Commander seats vs Admiral seats cleanly without double-running.
        super();
        installFastCombat(this.engine.container.combatService);

        // Stub starUpgradeService's DB repo so in-memory warp-gate / single-star
        // calls never touch Mongoose.
        const starUpgrade = this.engine.container.starUpgradeService as any;
        starUpgrade.gameRepo = {
            bulkWrite: async () => {},
            updateOne: async () => {},
        };

        this.tacticalExecutor = new TacticalExecutor(this.engine.container);

        // Rebind GameTickService._playAI to handle built-in AI seats, Commander
        // seats, and Admiral seats in deterministic seat order.
        const tick = this.engine.container.gameTickService as any;
        const rawPlayAI = Object.getPrototypeOf(tick)._playAI.bind(tick);
        this.basePlayAI = rawPlayAI;
        const ai = this.engine.container.aiService;

        tick._playAI = async (eventService: any, game: Game) => {
            await rawPlayAI(eventService, game);

            if (game !== this.game) return;

            for (const seat of this.seats.values()) {
                if (seat.player.defeated) continue;
                seat.player.lastSeen = new Date();

                if (this.admiralSeatSet.has(seat.index)) {
                    const dec = this.admiralDecisions.get(seat.index) ?? {};
                    if (dec.betray) await this._betray(seat);
                    else if (dec.court) await this._court(seat);

                    // Sync beliefs onto seat.decision so AgentEnv._fog applies them
                    // to hidden stars while TacticalExecutor runs.
                    seat.decision = {
                        ...seat.decision,
                        beliefs: dec.beliefs
                            ? { ships: dec.beliefs.ships }
                            : undefined,
                    };
                    const restore = this._fog(seat, true);
                    try {
                        const indVec = dec.beliefs?.industry;
                        const starVis = this.fog
                            ? (seat.memory.view?.stars ?? null)
                            : null;
                        await this.tacticalExecutor.executeTick(
                            game,
                            seat.player,
                            dec,
                            indVec
                                ? (i: number) => indVec[i] ?? undefined
                                : undefined,
                            starVis,
                        );
                    } finally {
                        restore();
                    }
                } else {
                    if (seat.decision.betray) await this._betray(seat);
                    else if (seat.decision.court) await this._court(seat);

                    const restore = this._fog(seat, true);
                    try {
                        await ai.playForAgent(eventService, game, seat.player);
                        if (seat.decision.explore) await this._explore(seat);
                    } finally {
                        restore();
                    }
                }
            }
        };
    }

    override async reset(options: AdmiralResetOptions): Promise<StepResult> {
        const admiralSeats = options.admiralSeats ?? [];
        const commanderSeats = options.commanderSeats ?? options.agents ?? [];
        const combined = Array.from(
            new Set([...admiralSeats, ...commanderSeats]),
        ).sort((a, b) => a - b);

        this.admiralSeatSet = new Set(admiralSeats);
        this.admiralDecisions.clear();
        for (const s of admiralSeats) {
            this.admiralDecisions.set(s, {});
        }

        return super.reset({
            ...options,
            agents: combined,
        });
    }

    override _needsDecision(seat: any): boolean {
        const tick = this.game.state.tick;
        const cycleStart =
            tick % this.game.settings.galaxy.productionTicks === 1;
        if (this.cycleOnly) return cycleStart;

        if (this.admiralSeatSet.has(seat.index)) {
            const attackKeys = this._visibleAttackKeys(seat);
            const newAttack = attackKeys
                .split(",")
                .some((k) => k && !seat.defendKeys.split(",").includes(k));
            seat.defendKeys = attackKeys;
            return (
                cycleStart ||
                newAttack ||
                tick - seat.lastDecisionTick >= this.maxGap
            );
        }

        return super._needsDecision(seat);
    }

    private _isFriendlyInboundCarrier(player: Player, car: any): boolean {
        if (car.isGift) return true;
        if (!car.ownedByPlayerId) return false;
        const pid = player._id.toString();
        if (car.ownedByPlayerId.toString() === pid) return true;
        const dip = this.engine.container.diplomacyService;
        if (dip.isFormalAlliancesEnabled(this.game)) {
            return (
                dip.getDiplomaticStatusToPlayer(
                    this.game,
                    player._id,
                    car.ownedByPlayerId,
                ).actualStatus === "allies"
            );
        }
        return false;
    }

    private _visibleAttackKeys(seat: any): string {
        const pid = seat.player._id.toString();
        const view = seat.memory.view;
        const keys = new Set<string>();
        for (const c of this.game.galaxy.carriers) {
            if (c.orbiting != null || !c.waypoints?.length) continue;
            if (this._isFriendlyInboundCarrier(seat.player, c)) continue;
            if (this.fog && view && !view.carriers.has(c)) continue;
            const destId = c.waypoints[0].destination.toString();
            const star = this.game.galaxy.stars.find(
                (s) => s._id.toString() === destId,
            );
            if (star?.ownedByPlayerId?.toString() === pid) {
                keys.add(`defend:${destId}`);
            }
        }
        return [...keys].sort().join(",");
    }

    private _computeOwnFacts(seat: any): AdmiralOwnFacts {
        const c = this.engine.container;
        const game = this.game;
        const player: Player = seat.player;
        const pid = player._id.toString();
        const stars = game.galaxy.stars;
        const S = stars.length;
        const starIndexById = new Map<string, number>();
        for (let i = 0; i < S; i++) {
            starIndexById.set(stars[i]._id.toString(), i);
        }

        const upgradeCost = (star: Star, type: InfrastructureType): number => {
            const key = game.settings.player.developmentCost[type];
            if (key === "none" || c.starDataService.isDeadStar(star)) return -1;
            const ic = c.infrastructureCostService;
            const fn =
                type === "economy"
                    ? ic.calculateEconomyCost.bind(ic)
                    : type === "industry"
                      ? ic.calculateIndustryCost.bind(ic)
                      : ic.calculateScienceCost.bind(ic);
            return (
                c.starUpgradeService._calculateUpgradeInfrastructureCost(
                    game,
                    star,
                    key,
                    type,
                    fn,
                ) ?? -1
            );
        };

        const warpCost = (star: Star): number => {
            const key = game.settings.specialGalaxy.warpgateCost;
            if (
                key === "none" ||
                star.warpGate ||
                c.starDataService.isDeadStar(star)
            ) {
                return -1;
            }
            const mult =
                game.constants.star.infrastructureExpenseMultipliers[key];
            const techs = c.technologyService.getStarEffectiveTechnologyLevels(
                game,
                star,
            );
            const tr = c.starService.calculateTerraformedResources(
                star,
                techs.terraforming,
            );
            const avg =
                c.infrastructureCostService.calculateAverageTerraformedResources(
                    tr,
                );
            return (
                c.infrastructureCostService.calculateWarpGateCost(
                    game,
                    mult,
                    avg,
                ) ?? -1
            );
        };

        const ecoCosts = new Array<number>(S).fill(-1);
        const indCosts = new Array<number>(S).fill(-1);
        const sciCosts = new Array<number>(S).fill(-1);
        const wgCosts = new Array<number>(S).fill(-1);

        for (let i = 0; i < S; i++) {
            if (stars[i].ownedByPlayerId?.toString() !== pid) continue;
            ecoCosts[i] = upgradeCost(stars[i], "economy");
            indCosts[i] = upgradeCost(stars[i], "industry");
            sciCosts[i] = upgradeCost(stars[i], "science");
            wgCosts[i] = warpCost(stars[i]);
        }

        const rps = c.researchService.researchProgressService;
        const researchProgress = this.technologies.map(
            (t) => (player.research as any)[t]?.progress ?? 0,
        );
        const researchRequired = this.technologies.map((t) => {
            const level = (player.research as any)[t]?.level ?? 1;
            if (!c.technologyService.isTechnologyResearchable(game, t as any)) {
                return 0;
            }
            return rps.getRequiredResearchProgress(game, t as any, level) ?? 0;
        });

        const budget = this._specialistBudget(player);
        const affordableStarSpecialists = this._starSpecialists()
            .filter((s) => this._specialistCost(s) <= budget)
            .map((s) => s.id);
        const affordableCarrierSpecialists = this._carrierSpecialists()
            .filter((s) => this._specialistCost(s) <= budget)
            .map((s) => s.id);

        const ownCarriers = game.galaxy.carriers
            .filter((car) => car.ownedByPlayerId?.toString() === pid)
            .map((car) => ({
                id: car._id.toString(),
                ships: car.ships ?? 0,
                orbiting: car.orbiting
                    ? (starIndexById.get(car.orbiting.toString()) ?? -1)
                    : -1,
                destination: car.waypoints?.length
                    ? (starIndexById.get(
                          car.waypoints[0].destination.toString(),
                      ) ?? -1)
                    : -1,
                delayTicks: car.waypoints?.length
                    ? (car.waypoints[0].delayTicks ?? 0)
                    : 0,
                specialist: car.specialistId ?? 0,
            }));

        const view = seat.memory.view;
        const attackMap = new Map<
            string,
            { starIndex: number; ticksUntil: number; ships: number }
        >();
        for (const car of game.galaxy.carriers) {
            if (car.orbiting != null || !car.waypoints?.length) continue;
            if (this._isFriendlyInboundCarrier(player, car)) continue;
            if (this.fog && view && !view.carriers.has(car)) continue;
            const wp = car.waypoints[0];
            const dstIdx = starIndexById.get(wp.destination.toString());
            if (dstIdx === undefined) continue;
            if (stars[dstIdx].ownedByPlayerId?.toString() !== pid) continue;
            const eta =
                c.waypointService.calculateWaypointTicksEta(game, car, wp) || 1;
            const ships =
                !this.fog || !view || view.carrierShips.has(car)
                    ? (car.ships ?? 1)
                    : 1;
            const k = `${dstIdx}:${eta}`;
            const prev = attackMap.get(k);
            if (prev) prev.ships += ships;
            else
                attackMap.set(k, { starIndex: dstIdx, ticksUntil: eta, ships });
        }

        const carrierMult =
            game.constants.star.infrastructureExpenseMultipliers[
                game.settings.specialGalaxy.carrierCost
            ];
        const carrierCost = c.infrastructureCostService.calculateCarrierCost(
            game,
            carrierMult,
        );

        return {
            credits: player.credits ?? 0,
            specialistBudget: budget,
            carrierCost,
            researchingNow: this.technologies.indexOf(player.researchingNow),
            researchingNext: this.technologies.indexOf(
                (player.researchQueue?.[0] ?? "") as string,
            ),
            researchProgress,
            researchRequired,
            affordableStarSpecialists,
            affordableCarrierSpecialists,
            upgradeCosts: {
                economy: ecoCosts,
                industry: indCosts,
                science: sciCosts,
                warpGate: wgCosts,
            },
            ownCarriers,
            incomingAttacks: [...attackMap.values()],
        };
    }

    observeAdmiral(
        index: number,
        options: { truthFrame?: boolean } = {},
    ): AdmiralObservation {
        const seat = this._seat(index);
        const prodTicks = this.game.settings.galaxy.productionTicks;
        const tick = this.game.state.tick;

        const obs: AdmiralObservation = {
            seat: index,
            tick,
            cycle: Math.floor(tick / prodTicks),
            ticksToProduction: prodTicks - (tick % prodTicks),
            ticksToEnd: this.game.state.ticksToEnd ?? null,
            defeated: seat.player.defeated,
            frame: seatFrame(this.game, seat.memory, this.technologies),
            ownFacts: this._computeOwnFacts(seat),
        };

        if (options.truthFrame) {
            const truthMem = new SeatMemory(this.game.galaxy.stars.length);
            truthMem.update(
                this.engine,
                this.game,
                seat.player,
                fullView(this.game),
            );
            obs.truthFrame = seatFrame(this.game, truthMem, this.technologies);
            obs.truthFacts = {
                playerCredits: this.game.galaxy.players.map(
                    (p) => p.credits ?? 0,
                ),
                playerSpecTokens: this.game.galaxy.players.map(
                    (p) => p.creditsSpecialists ?? 0,
                ),
                playerResearchingNow: this.game.galaxy.players.map((p) =>
                    this.technologies.indexOf(p.researchingNow),
                ),
                playerResearchProgress: this.game.galaxy.players.map((p) =>
                    this.technologies.map(
                        (t) => (p.research as any)[t]?.progress ?? 0,
                    ),
                ),
            };
        }

        return obs;
    }

    async actAdmiral(index: number, decision: AdmiralDecision) {
        const seat = this._seat(index);
        this.admiralSeatSet.add(index);
        this.admiralDecisions.set(index, decision);
        seat.lastDecisionTick = this.game.state.tick;

        for (const { star, specialist } of decision.hire ?? []) {
            const target = this.game.galaxy.stars[star];
            if (!target) continue;
            try {
                await this.engine.container.specialistHireService.hireStarSpecialist(
                    this.game,
                    seat.player,
                    target._id,
                    specialist,
                    this.engine.container.statisticsService,
                );
                seat.hired += 1;
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
                seat.hiresRefused += 1;
            }
        }

        if (decision.research) {
            const tech = decision.research as any;
            const technology = this.engine.container.technologyService;
            if (technology.isTechnologyResearchable(this.game, tech)) {
                seat.player.researchingNow = tech;
                seat.player.researchQueue = [
                    (decision.researchNext ?? "random") as any,
                ];
            }
        }
    }

    // Build a fog-legal determinized copy of `this.game` from `seat`'s view +
    // a conservation-consistent `BeliefWorld`. Hidden enemy carriers, hidden
    // carrier waypoints, hidden star truth, and hidden enemy credits/progress
    // are NEVER copied from `this.game`.
    private _buildDeterminizedWorld(seat: any, world: BeliefWorld): Game {
        const pid = seat.player._id.toString();
        const view = seat.memory.view!;
        const copy = this.engine.clone(this.game);
        const origStars = this.game.galaxy.stars;
        const origCarriers = this.game.galaxy.carriers;

        // 1. Replace hidden star ships & infrastructure with believed values.
        copy.galaxy.stars.forEach((s, i) => {
            if (s.ownedByPlayerId?.toString() === pid) return;
            if (view.stars[i] !== VISIBLE) {
                const ships = Math.max(
                    0,
                    Math.round(world.ships[i] ?? seat.memory.ships[i] ?? 0),
                );
                s.ships = ships;
                s.shipsActual = ships;
            }
            if (view.stars[i] === HIDDEN) {
                s.infrastructure = {
                    economy: Math.max(
                        0,
                        Math.round(
                            world.economy?.[i] ??
                                seat.memory.infrastructure[i * 3],
                        ),
                    ),
                    industry: Math.max(
                        0,
                        Math.round(
                            world.industry?.[i] ??
                                seat.memory.infrastructure[i * 3 + 1],
                        ),
                    ),
                    science: Math.max(
                        0,
                        Math.round(
                            world.science?.[i] ??
                                seat.memory.infrastructure[i * 3 + 2],
                        ),
                    ),
                };
                s.warpGate = Boolean(seat.memory.warpGate[i]);
                if (!seat.memory.specialist[i]) {
                    s.specialistId = null;
                }
            }
            if (
                origStars[i].isNebula &&
                s.ownedByPlayerId?.toString() !== pid
            ) {
                // Nebulas hide natural resources unless owned.
                s.naturalResources = { economy: 25, industry: 25, science: 25 };
            }
        });

        // 2. Keep ONLY own carriers and enemy carriers currently in scanning range;
        //    truncate enemy carrier waypoints to the first hop and scrub hidden ship counts.
        const keptCarriers: any[] = [];
        for (let idx = 0; idx < origCarriers.length; idx++) {
            const origCar = origCarriers[idx];
            const copyCar = copy.galaxy.carriers[idx];
            if (origCar.ownedByPlayerId?.toString() === pid) {
                keptCarriers.push(copyCar);
                continue;
            }
            if (!view.carriers.has(origCar)) continue;
            if (copyCar.waypoints.length > 1) {
                copyCar.waypoints = copyCar.waypoints.slice(0, 1);
                copyCar.waypointsLooped = false;
            }
            if (!view.carrierShips.has(origCar)) {
                copyCar.ships = 10; // Neutral estimate for Scrambler/Nebula carrier
            }
            keptCarriers.push(copyCar);
        }
        copy.galaxy.carriers = keptCarriers;

        // 3. Scrub hidden opponent credits/progress and hand all non-viewer seats
        //    to the fast built-in AI during the lookahead rollout so private
        //    opponent policies/memories are never leaked.
        for (const p of copy.galaxy.players) {
            if (p._id.toString() === pid) continue;
            p.credits = p.defeated ? 0 : 100;
            p.creditsSpecialists = 0;
            p.userId = null;
            for (const t of this.technologies) {
                if ((p.research as any)[t]) {
                    (p.research as any)[t].progress = 0;
                }
            }
        }

        return copy;
    }

    // Fog-legal 1-cycle (or N-tick) lookahead planner:
    // Evaluates `plans` (M candidate AdmiralDecisions) across `worlds` (K
    // conservation-consistent belief worlds), rolling each (plan, world) forward
    // `ticks` ticks with identical RNG seed per world so plans are compared on
    // paired common random numbers.
    async lookaheadAdmiral(
        index: number,
        plans: AdmiralDecision[],
        worlds: BeliefWorld[],
        ticks: number,
        seed = 1,
    ) {
        const seat = this._seat(index);
        const origGame = this.game;
        const seedState = this.engine.saveSeedState();
        const origSeats = new Map(this.seats);
        const origAdmiralSet = new Set(this.admiralSeatSet);
        const origAdmiralDecs = new Map(this.admiralDecisions);

        const savedSeatState = {
            player: seat.player,
            decision: seat.decision,
            memory: seat.memory,
            hired: seat.hired,
            hiresRefused: seat.hiresRefused,
            courting: seat.courting,
            courtSpent: seat.courtSpent,
            givenUp: new Set(seat.givenUp),
            gifted: seat.gifted,
            alliedTick: seat.alliedTick,
        };

        const out: {
            frame: any;
            ownFacts: AdmiralOwnFacts;
            players: ReturnType<AgentEnv["standings"]>["players"];
        }[][] = [];

        try {
            const worldTemplates = worlds.map((w) =>
                this.engine.toJSON(this._buildDeterminizedWorld(seat, w)),
            );

            for (const plan of plans) {
                const planResults: {
                    frame: any;
                    ownFacts: AdmiralOwnFacts;
                    players: ReturnType<AgentEnv["standings"]>["players"];
                }[] = [];

                for (let wi = 0; wi < worldTemplates.length; wi++) {
                    const copy = this.engine.fromJSON(
                        JSON.parse(JSON.stringify(worldTemplates[wi])),
                    );
                    const copyPlayer = copy.galaxy.players[index];
                    const byId = new Map(
                        copy.galaxy.players.map((p) => [p._id.toString(), p]),
                    );

                    // Isolate `this.seats` to ONLY the planning seat during rollout
                    // so every opponent is played by the built-in AI on the determinized state.
                    this.seats = new Map([[index, seat]]);
                    this.admiralSeatSet = new Set([index]);
                    this.admiralDecisions = new Map([[index, plan]]);

                    seat.player = copyPlayer;
                    seat.memory = savedSeatState.memory.copy();
                    seat.decision = {};
                    seat.hired = savedSeatState.hired;
                    seat.hiresRefused = savedSeatState.hiresRefused;
                    seat.courting = savedSeatState.courting
                        ? (byId.get(savedSeatState.courting._id.toString()) ??
                          null)
                        : null;
                    seat.courtSpent = savedSeatState.courtSpent;
                    seat.givenUp = new Set(
                        [...savedSeatState.givenUp]
                            .map((p) => byId.get(p._id.toString()))
                            .filter((p): p is Player => p != null),
                    );
                    seat.gifted = savedSeatState.gifted;
                    seat.alliedTick = savedSeatState.alliedTick;

                    this.game = copy;
                    this.engine.seed(seed + wi * 997, 0x61);

                    await this.actAdmiral(index, plan);
                    const endTick = Math.min(
                        copy.state.tick + ticks,
                        this.maxTicks,
                    );
                    while (!copy.state.endDate && copy.state.tick < endTick) {
                        await this.engine.tick(copy, 1);
                        seat.memory.update(this.engine, copy, copyPlayer);
                    }

                    planResults.push({
                        frame: seatFrame(copy, seat.memory, this.technologies),
                        ownFacts: this._computeOwnFacts(seat),
                        players: this.standings().players,
                    });
                }

                out.push(planResults);
            }
        } finally {
            this.game = origGame;
            this.engine.restoreSeedState(seedState);
            this.seats = origSeats;
            this.admiralSeatSet = origAdmiralSet;
            this.admiralDecisions = origAdmiralDecs;

            seat.player = savedSeatState.player;
            seat.decision = savedSeatState.decision;
            seat.memory = savedSeatState.memory;
            seat.hired = savedSeatState.hired;
            seat.hiresRefused = savedSeatState.hiresRefused;
            seat.courting = savedSeatState.courting;
            seat.courtSpent = savedSeatState.courtSpent;
            seat.givenUp = savedSeatState.givenUp;
            seat.gifted = savedSeatState.gifted;
            seat.alliedTick = savedSeatState.alliedTick;
        }

        return out;
    }
}
