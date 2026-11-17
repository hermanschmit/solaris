// An environment for agents playing seats of a simulated game.
//
// The agent is a commander: it scores the orders the built-in AI proposes
// (defend, invade or claim a star), tunes the built-in AI's config (for
// example the spending split) and picks research. The built-in AI's
// executor then carries the orders out: it finds ships, buys carriers and
// writes waypoints. Seats without an agent are played by the built-in AI.
//
// The agent is asked for a decision at the start of each cycle, when a new
// attack on one of its stars appears, and at least every maxGap ticks. It
// decides on the state at the end of a tick, and its executor acts on that
// decision during the next tick, at the point where the built-in AI plays.
// The decision stays in force until the next one. An order that first
// appears during a tick is only acted on once the agent has scored it, so
// the agent reacts to it one tick later than the built-in AI would.
import fs from "fs";
import { Specialist, ValidationError } from "@solaris/common";
import mongoose from "mongoose";
import { SimEngine } from "./engine";
import { Game } from "../services/types/Game";
import { Player } from "../services/types/Player";
import { AiConfig } from "../services/types/Ai";
import { AiOrderCandidate } from "../services/ai";
import { SeatMemory, seatFrame, applyFog, fullView } from "./fog";
import { BeliefRecorder } from "./beliefData";

export interface AgentDecision {
    // Score per order key; orders left out are dropped. Omit to keep the
    // built-in AI's choice of orders and scores.
    orders?: Record<string, number>;
    // Multiplies the built-in AI's own score for an order, by order key, as
    // the executor computes it when it acts. Orders left out keep their
    // score, so all weights of 1 play exactly like the built-in AI.
    orderWeights?: Record<string, number>;
    // Built-in AI tuning for this seat. Fields left out keep their
    // defaults. Without config, the seat keeps its current tuning.
    config?: Partial<AiConfig>;
    // Technology to research now and next.
    research?: string;
    // Star specialists to hire now, with specialist tokens (or credits,
    // as the game's settings say): specialist id at a star index the seat
    // owns. A hire the game refuses (unaffordable, or not allowed there) is
    // skipped.
    hire?: { star: number; specialist: number }[];
    // Carrier specialists to hire now, the same way: specialist id on a
    // carrier (id, as ownCarriers lists it) orbiting a star the seat owns.
    hireCarrier?: { carrier: string; specialist: number }[];
    // Under fog: what the agent believes, for the executor to act on. ships
    // gives believed ships per star index, used where the seat cannot see
    // a star's ships. Stars left out (or null) show the last ships seen.
    beliefs?: { ships?: (number | null)[] };
    // Keep an Explorer carrier claiming unowned stars beyond the seat's
    // own hyperspace range (see AgentEnv._explore).
    explore?: boolean;
    // Buy an alliance with the nearest built-in AI player in scanning
    // range: declare it an ally and gift technology or credits, whichever
    // buys reputation cheapest, until its reputation reaches the level at
    // which the AI declares back (see AgentEnv._court).
    court?: boolean;
    // Break every alliance with a built-in AI player (declare it an enemy).
    betray?: boolean;
}

export interface ResetOptions {
    settings?: string;
    seed?: number;
    // Seat indexes played by agents.
    agents: number[];
    // Built-in AI tuning for other seats, by seat index.
    aiConfigs?: Record<number, Partial<AiConfig>>;
    maxTicks?: number;
    maxGap?: number;
    // Agent seats see only what their scanners show: their observations
    // and their executor (the built-in AI acting on their orders) both
    // work from the seat's view and memory instead of the whole game.
    fog?: boolean;
    // Record the game for belief-model training (see beliefData.ts) and
    // save it to this file when the game ends.
    record?: string;
    // Stored in the recording's header.
    recordInfo?: Record<string, any>;
}

export interface Observation {
    seat: number;
    tick: number;
    cycle: number;
    // Ticks until the next production (end of cycle).
    ticksToProduction: number;
    defeated: boolean;
    // With starIndex, the star's index in the galaxy's star list.
    candidates: (AiOrderCandidate & { starIndex: number })[];
    // The seat's fog-of-war view, as GET /api/game/:gameId/galaxy returns
    // it. Only when asked for, since it is slower to build.
    view?: any;
    // When asked for: what the seat sees and remembers, in the
    // form sim/python/belief_data.py builds belief-model features from.
    frame?: any;
    // When asked for: the same, as if the seat saw the whole galaxy. For
    // a privileged critic in training, never for the policy.
    truthFrame?: any;
    // What the seat can spend on specialists, and the star specialists it
    // can afford now (ids, among the header's starSpecialists).
    specialistBudget: number;
    affordableStarSpecialists: number[];
    // The same for carrier specialists, and the seat's own carriers: id,
    // ships, the star index it orbits (-1 in transit) and its specialist
    // (0 for none).
    affordableCarrierSpecialists: number[];
    ownCarriers: {
        id: string;
        ships: number;
        orbiting: number;
        // The star it is flying to next, -1 when it has no waypoints.
        destination: number;
        specialist: number;
    }[];
    // The seat's credits.
    credits: number;
}

export interface StepResult {
    tick: number;
    done: boolean;
    // Seats that need a decision before the next step.
    seats: number[];
}

interface Seat {
    index: number;
    player: Player;
    decision: AgentDecision;
    lastDecisionTick: number;
    // Specialists hired, and hires the game refused.
    hired: number;
    hiresRefused: number;
    defendKeys: string;
    memory: SeatMemory;
    // The AI player being courted, credits spent on gifts, and the tick
    // the alliance formed (null while it has not).
    courting: Player | null;
    courtSpent: number;
    givenUp: Set<Player>;
    gifted: number;
    alliedTick: number | null;
}

const EXPLORER = 6; // carrier specialist: +3 hyperspace, faster, weaker
// ReputationService: the AI declares a player an ally at this reputation,
// which each gift raises by REPUTATION_INCREMENT (credits) or by the levels
// given (technology).
const ALLY_REPUTATION = 5;
const REPUTATION_INCREMENT = 1;
// Credits a seat spends courting one player before trying another.
const COURT_BUDGET = 1000;

export class AgentEnv {
    engine = new SimEngine();
    game!: Game;
    seats = new Map<number, Seat>();
    maxTicks = 3000;
    maxGap = 4;
    fog = false;
    recorder: BeliefRecorder | null = null;
    recordFile: string | null = null;
    technologies: string[] = [];

    constructor() {
        const ai = this.engine.container.aiService;
        const tick = this.engine.container.gameTickService as any;
        const playAI = tick._playAI.bind(tick);

        // Agents' executors run inside the tick, right after the built-in AI
        // players, so later steps of the tick (research, waypoint culling,
        // the win check) treat them exactly as they treat the built-in AI.
        tick._playAI = async (eventService: any, game: Game) => {
            await playAI(eventService, game);

            if (game !== this.game) return;

            for (const seat of this.seats.values()) {
                if (seat.player.defeated) continue;

                seat.player.lastSeen = new Date();

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
        };

        ai.getOrderScores = (game, player) => {
            const seat = this._seatOf(player);
            const orders = seat?.decision.orders;
            return orders ? new Map(Object.entries(orders)) : null;
        };

        ai.getOrderWeights = (game, player) => {
            const weights = this._seatOf(player)?.decision.orderWeights;
            return weights ? new Map(Object.entries(weights)) : null;
        };
    }

    // Court the nearest built-in AI player in scanning range (the trade
    // setting requires it): declare it an ally, and gift technology or
    // credits, whichever is cheapest per point of its reputation for the
    // seat, until the reputation reaches the AI's alliance threshold and it
    // declares back. Each gift that raises reputation lets the AI send a
    // technology back, as TradeService.tryTradeBack does. The trade service
    // itself writes to the database, so the gifts are applied here in
    // memory, through the same reputation calls.
    async _court(seat: Seat) {
        const c = this.engine.container as any;
        const game = this.game;
        const player = seat.player;
        const agents = new Set([...this.seats.values()].map((s) => s.player));
        const allied = (other: Player) =>
            c.diplomacyService.getDiplomaticStatusToPlayer(
                game,
                player._id,
                other._id,
            ).actualStatus === "allies";

        let target = seat.courting;
        if (target && seat.courtSpent > COURT_BUDGET && !allied(target)) {
            // Fighting keeps resetting its reputation: try someone else.
            seat.givenUp.add(target);
            target = null;
        }
        if (!target || target.defeated) {
            seat.courtSpent = 0;
            const own = game.galaxy.stars.filter(
                (s) => s.ownedByPlayerId?.toString() === player._id.toString(),
            );
            let best = Infinity;
            target = null;
            for (const other of game.galaxy.players) {
                if (
                    agents.has(other) ||
                    other.defeated ||
                    seat.givenUp.has(other)
                )
                    continue;
                if (
                    !c.scanningService.isInScanningRangeOfPlayer(
                        game,
                        player,
                        other,
                    )
                )
                    continue;
                const id = other._id.toString();
                for (const star of game.galaxy.stars) {
                    if (star.ownedByPlayerId?.toString() !== id) continue;
                    for (const mine of own) {
                        const d = c.distanceService.getDistanceBetweenLocations(
                            mine.location,
                            star.location,
                        );
                        if (d < best) {
                            best = d;
                            target = other;
                        }
                    }
                }
            }
            seat.courting = target;
        }
        if (!target) return;

        if (
            c.diplomacyService.getDiplomaticStatusToPlayer(
                game,
                player._id,
                target._id,
            ).statusTo !== "allies"
        ) {
            try {
                await c.diplomacyService.declareAlly(
                    c.eventService,
                    game,
                    player._id,
                    target._id,
                    false,
                );
            } catch {
                return; // the alliance cap
            }
        }

        const reputation = () =>
            c.reputationService.getReputation(target, player).reputation.score;

        for (let k = 0; k < 10 && reputation() < ALLY_REPUTATION; k++) {
            if (
                !c.scanningService.isInScanningRangeOfPlayer(
                    game,
                    player,
                    target,
                )
            )
                break;
            const creditsCost = Math.max(
                1,
                (c.playerStatisticsService.getStats(game, target).totalEconomy *
                    10) /
                    2,
            );
            let gift: { cost: number; points: number; tech?: any } = {
                cost: creditsCost,
                points: REPUTATION_INCREMENT,
            };
            for (const tech of c.tradeService.listTradeableTechnologies(
                game,
                player,
                target._id,
            )) {
                const points = tech.level - target.research[tech.name].level;
                if (tech.cost / points < gift.cost / gift.points)
                    gift = { cost: tech.cost, points, tech };
            }
            if (player.credits < gift.cost) break;

            if (gift.tech) {
                this._giveTechnology(player, target, gift.tech);
            } else {
                player.credits -= gift.cost;
                target.credits += gift.cost;
            }
            seat.gifted += gift.cost;
            seat.courtSpent += gift.cost;
            const rep = await c.reputationService.increaseReputation(
                c.eventService,
                game,
                target,
                player,
                gift.points,
                false,
            );

            // The AI trades back a random technology it can afford.
            if (rep.reputation.score >= 1) {
                const back = c.tradeService
                    .listTradeableTechnologies(game, target, player._id)
                    .filter((t: any) => t.cost <= target!.credits);
                if (back.length) {
                    const tech = back[Math.floor(Math.random() * back.length)];
                    const points =
                        tech.level - player.research[tech.name].level;
                    this._giveTechnology(target, player, tech);
                    await c.reputationService.increaseReputation(
                        c.eventService,
                        game,
                        player,
                        target,
                        points,
                        false,
                    );
                }
            }
        }

        if (seat.alliedTick === null && allied(target)) {
            seat.alliedTick = game.state.tick;
        }
    }

    async _betray(seat: Seat) {
        const c = this.engine.container as any;
        const agents = new Set([...this.seats.values()].map((s) => s.player));
        for (const other of this.game.galaxy.players) {
            if (other === seat.player || agents.has(other)) continue;
            const status = c.diplomacyService.getDiplomaticStatusToPlayer(
                this.game,
                seat.player._id,
                other._id,
            );
            if (status.statusTo === "allies") {
                await c.diplomacyService.declareEnemy(
                    c.eventService,
                    this.game,
                    seat.player._id,
                    other._id,
                    false,
                );
            }
        }
    }

    _giveTechnology(from: Player, to: Player, tech: any) {
        from.credits -= tech.cost;
        to.research[tech.name].level = tech.level;
        to.research[tech.name].progress = 0;
    }

    // An Explorer (+3 hyperspace) carrier claims unowned stars the seat's
    // executor cannot reach. With none, hire one on a carrier at the own
    // star closest to such a star (building the carrier if need be); an
    // idle one flies to the nearest unowned star in its range. Star owners
    // are public, so this needs nothing hidden by fog.
    async _explore(seat: Seat) {
        const c = this.engine.container as any;
        const game = this.game;
        const player = seat.player;
        const id = player._id.toString();
        const distance = (a: any, b: any) =>
            c.distanceService.getDistanceBetweenLocations(
                a.location,
                b.location,
            );
        const level = player.research.hyperspace.level;
        const near = c.distanceService.getHyperspaceDistance(game, level);
        const far = c.distanceService.getHyperspaceDistance(game, level + 3);

        const owned = game.galaxy.stars.filter(
            (s) => s.ownedByPlayerId?.toString() === id,
        );
        const free = game.galaxy.stars.filter(
            (s) => !s.ownedByPlayerId && !c.starDataService.isDeadStar(s),
        );
        if (!owned.length || !free.length) return;

        const explorers = game.galaxy.carriers.filter(
            (x) =>
                x.ownedByPlayerId?.toString() === id &&
                x.specialistId === EXPLORER,
        );
        const taken = new Set(
            explorers.flatMap((x) =>
                (x.waypoints ?? []).map((w) => w.destination.toString()),
            ),
        );

        for (const carrier of explorers) {
            if (!carrier.orbiting || carrier.waypoints?.length) continue;
            const from = game.galaxy.stars.find(
                (s) => s._id.toString() === carrier.orbiting!.toString(),
            )!;
            const options = free
                .filter((s) => !taken.has(s._id.toString()))
                .map((s) => [distance(from, s), s] as const)
                .filter(([d]) => d <= far)
                .sort((a, b) => a[0] - b[0]);
            if (!options.length) continue;
            const target = options[0][1];
            try {
                await c.saveWaypointsService.saveWaypointsForCarrier(
                    game,
                    player,
                    carrier,
                    [
                        {
                            _id: new mongoose.Types.ObjectId(),
                            source: from._id,
                            destination: target._id,
                            action: "nothing",
                            actionShips: 0,
                            delayTicks: 0,
                        },
                    ],
                    false,
                    false,
                );
                taken.add(target._id.toString());
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
            }
        }

        if (explorers.length) return;

        // Stars only an Explorer reaches, and the own star closest to one.
        let best: { d: number; star: any } | null = null;
        for (const target of free) {
            let closest: { d: number; star: any } | null = null;
            for (const star of owned) {
                const d = distance(star, target);
                if (!closest || d < closest.d) closest = { d, star };
            }
            if (closest && closest.d > near && closest.d <= far) {
                if (!best || closest.d < best.d) best = closest;
            }
        }
        if (!best) return;

        const explorer = this._carrierSpecialists().find(
            (x) => x.id === EXPLORER,
        );
        if (
            !explorer ||
            this._specialistCost(explorer) > this._specialistBudget(player)
        ) {
            return;
        }

        try {
            let carrier = game.galaxy.carriers.find(
                (x) =>
                    x.ownedByPlayerId?.toString() === id &&
                    x.orbiting?.toString() === best!.star._id.toString() &&
                    !x.waypoints?.length &&
                    x.specialistId == null,
            );
            if (!carrier) {
                if ((best.star.ships ?? 0) < 1) return;
                const built = await c.starUpgradeService.buildCarrier(
                    game,
                    player,
                    best.star._id,
                    1,
                    false,
                    c.statisticsService,
                );
                carrier = c.carrierService.getById(game, built.carrier._id);
            }
            await c.specialistHireService.hireCarrierSpecialist(
                game,
                player,
                carrier!._id,
                EXPLORER,
                c.statisticsService,
            );
            seat.hired += 1;
        } catch (e: any) {
            if (!(e instanceof ValidationError)) throw e;
            seat.hiresRefused += 1;
        }
    }

    _seatOf(player: Player): Seat | undefined {
        for (const seat of this.seats.values()) {
            if (seat.player._id.toString() === player._id.toString()) {
                return seat;
            }
        }
    }

    async reset(options: ResetOptions): Promise<StepResult> {
        const seed = options.seed ?? 1;
        this.maxTicks = options.maxTicks ?? 3000;
        this.maxGap = options.maxGap ?? 4;
        this.fog = options.fog ?? false;
        this.engine.aiConfigs.clear();
        this.seats.clear();

        this.engine.seed(seed);
        this.game = await this.engine.createGame(
            this.engine.loadSettings(options.settings ?? "standard"),
        );
        const players = this.game.galaxy.players;

        for (const [index, config] of Object.entries(options.aiConfigs ?? {})) {
            this.engine.setAiConfig(this.game, players[+index]._id, config);
        }

        for (const index of options.agents) {
            const player = players[index];
            if (!player) throw new Error(`There is no seat ${index}.`);

            this.engine.setExternallyControlled(
                this.game,
                player._id,
                new mongoose.Types.ObjectId(),
            );
            player.lastSeen = new Date();
            this.seats.set(index, {
                index,
                player,
                decision: {},
                lastDecisionTick: 0,
                hired: 0,
                hiresRefused: 0,
                defendKeys: "",
                memory: new SeatMemory(this.game.galaxy.stars.length),
                courting: null,
                courtSpent: 0,
                givenUp: new Set(),
                gifted: 0,
                alliedTick: null,
            });
        }

        const recorder = new BeliefRecorder(this.engine, this.game, {
            settings: options.settings ?? "standard",
            seed,
            agents: options.agents,
            ...options.recordInfo,
        });
        this.technologies = recorder.technologies;
        this.recorder = options.record ? recorder : null;
        this.recordFile = options.record ?? null;
        this.recorder?.capture(this.game);
        this._updateMemories();

        return this._result([...this.seats.keys()]);
    }

    _updateMemories() {
        for (const seat of this.seats.values()) {
            if (!seat.player.defeated) {
                seat.memory.update(this.engine, this.game, seat.player);
            }
        }
    }

    // Under fog, make the game show only what the seat knows; returns the
    // function that puts it back. refresh first rescans, for use mid-tick.
    _fog(seat: Seat, refresh = false): () => void {
        if (!this.fog) return () => {};
        if (refresh) seat.memory.update(this.engine, this.game, seat.player);

        const ships = seat.decision.beliefs?.ships;
        return applyFog(
            this.game,
            seat.player,
            seat.memory,
            (i) => ships?.[i] ?? undefined,
        );
    }

    _proposeOrders(seat: Seat): AiOrderCandidate[] {
        const restore = this._fog(seat);
        try {
            return this.engine.container.aiService.proposeOrders(
                this.game,
                seat.player,
            );
        } finally {
            restore();
        }
    }

    // The game's constant facts (star positions, ranges, technologies), as
    // sim/python/belief_data.py's Static takes them.
    header() {
        const {
            stars,
            players,
            productionTicks,
            carrierSpeed,
            technologies,
            hyperspaceRange,
            scanningRange,
        } = new BeliefRecorder(this.engine, this.game).header;
        return {
            stars,
            players,
            productionTicks,
            carrierSpeed,
            technologies,
            hyperspaceRange,
            scanningRange,
            starSpecialists: this._starSpecialists().map((s) => ({
                id: s.id,
                name: s.name,
            })),
        };
    }

    // Star specialists that can be hired in this game, by id.
    _starSpecialists() {
        const { specialistService, specialistBanService } =
            this.engine.container;
        if (this.game.settings.specialGalaxy.specialistCost === "none") {
            return [];
        }
        return specialistService
            .listStar(this.game)
            .filter(
                (s) =>
                    !specialistBanService.isStarSpecialistBanned(
                        this.game,
                        s.id,
                    ),
            )
            .sort((a, b) => a.id - b.id);
    }

    // Carrier specialists that can be hired in this game, by id.
    _carrierSpecialists() {
        const { specialistService, specialistBanService } =
            this.engine.container;
        if (this.game.settings.specialGalaxy.specialistCost === "none") {
            return [];
        }
        return specialistService
            .listCarrier(this.game)
            .filter(
                (s) =>
                    !specialistBanService.isCarrierSpecialistBanned(
                        this.game,
                        s.id,
                    ),
            )
            .sort((a, b) => a.id - b.id);
    }

    _specialistCost(specialist: Specialist) {
        const currency = this.game.settings.specialGalaxy.specialistsCurrency;
        return this.engine.container.specialistService.getSpecialistActualCost(
            this.game,
            specialist,
        )[currency];
    }

    _specialistBudget(player: Player) {
        const currency = this.game.settings.specialGalaxy.specialistsCurrency;
        return (player as any)[currency] ?? 0;
    }

    _isDone() {
        return (
            Boolean(this.game.state.endDate) ||
            this.game.state.tick >= this.maxTicks
        );
    }

    _result(seats: number[]): StepResult {
        return { tick: this.game.state.tick, done: this._isDone(), seats };
    }

    async observe(
        index: number,
        options: { view?: boolean; frame?: boolean; truthFrame?: boolean } = {},
    ) {
        const seat = this._seat(index);
        const ticks = this.game.settings.galaxy.productionTicks;
        const tick = this.game.state.tick;

        const starIndex = (id: any) =>
            this.game.galaxy.stars.findIndex(
                (s) => s._id.toString() === id.toString(),
            );
        const observation: Observation = {
            seat: index,
            tick,
            cycle: Math.floor(tick / ticks),
            ticksToProduction: ticks - (tick % ticks),
            defeated: seat.player.defeated,
            candidates: this._proposeOrders(seat).map((c) => ({
                ...c,
                starIndex: this.game.galaxy.stars.findIndex(
                    (s) => s._id.toString() === c.star,
                ),
            })),
            specialistBudget: this._specialistBudget(seat.player),
            affordableStarSpecialists: this._starSpecialists()
                .filter(
                    (s) =>
                        this._specialistCost(s) <=
                        this._specialistBudget(seat.player),
                )
                .map((s) => s.id),
            affordableCarrierSpecialists: this._carrierSpecialists()
                .filter(
                    (s) =>
                        this._specialistCost(s) <=
                        this._specialistBudget(seat.player),
                )
                .map((s) => s.id),
            ownCarriers: this.game.galaxy.carriers
                .filter(
                    (c) =>
                        c.ownedByPlayerId?.toString() ===
                        seat.player._id.toString(),
                )
                .map((c) => ({
                    id: c._id.toString(),
                    ships: c.ships ?? 0,
                    orbiting: c.orbiting
                        ? this.game.galaxy.stars.findIndex(
                              (s) =>
                                  s._id.toString() === c.orbiting!.toString(),
                          )
                        : -1,
                    destination: c.waypoints?.length
                        ? starIndex(c.waypoints[0].destination)
                        : -1,
                    specialist: c.specialistId ?? 0,
                })),
            credits: seat.player.credits ?? 0,
        };

        if (options.frame) {
            observation.frame = seatFrame(
                this.game,
                seat.memory,
                this.technologies,
            );
        }

        if (options.truthFrame) {
            const memory = new SeatMemory(this.game.galaxy.stars.length);
            memory.update(
                this.engine,
                this.game,
                seat.player,
                fullView(this.game),
            );
            observation.truthFrame = seatFrame(
                this.game,
                memory,
                this.technologies,
            );
        }

        if (options.view) {
            observation.view = await this.engine.viewAs(
                this.game,
                seat.player.userId!,
            );
        }

        return observation;
    }

    async act(index: number, decision: AgentDecision) {
        const seat = this._seat(index);
        seat.decision = decision;
        seat.lastDecisionTick = this.game.state.tick;

        if (decision.config) {
            this.engine.setAiConfig(
                this.game,
                seat.player._id,
                decision.config,
            );
        }

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

        for (const { carrier, specialist } of decision.hireCarrier ?? []) {
            const target = this.game.galaxy.carriers.find(
                (c) => c._id.toString() === carrier,
            );
            if (!target) continue;
            try {
                await this.engine.container.specialistHireService.hireCarrierSpecialist(
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

            if (!technology.isTechnologyResearchable(this.game, tech)) {
                throw new Error(`${tech} is not researchable in this game.`);
            }

            seat.player.researchingNow = tech;
            seat.player.researchingNext = tech;
        }
    }

    // Try other order weights for one seat on copies of the game: for each
    // plan, play a copy forward `ticks` ticks with the seat's decision's
    // order weights replaced by the plan's, everyone else playing as they
    // are, and report where every player stands then. The game itself, the
    // seats' memories and the random numbers it would have drawn are left
    // as they were. Each copy draws the same random numbers (from `seed`),
    // so plans differ only by what they do.
    async lookahead(
        index: number,
        plans: Record<string, number>[],
        ticks: number,
        seed = 1,
    ) {
        const seat = this._seat(index);
        const game = this.game;
        const random = Math.random;
        const saved = [...this.seats.values()].map((s) => ({
            seat: s,
            player: s.player,
            decision: s.decision,
            memory: s.memory,
        }));

        const results: ReturnType<AgentEnv["standings"]>["players"][] = [];
        try {
            for (const plan of plans) {
                const copy = this.engine.clone(game);
                const byId = new Map(
                    copy.galaxy.players.map((p) => [p._id.toString(), p]),
                );
                for (const s of saved) {
                    s.seat.player = byId.get(s.player._id.toString())!;
                    s.seat.memory = s.memory.copy();
                    s.seat.decision = s.decision;
                }
                seat.decision = { ...seat.decision, orderWeights: plan };

                this.game = copy;
                this.engine.seed(seed);
                const end = Math.min(game.state.tick + ticks, this.maxTicks);
                while (!copy.state.endDate && copy.state.tick < end) {
                    await this.engine.tick(copy, 1);
                }
                results.push(this.standings().players);
            }
        } finally {
            this.game = game;
            Math.random = random;
            for (const s of saved) {
                s.seat.player = s.player;
                s.seat.decision = s.decision;
                s.seat.memory = s.memory;
            }
        }
        return results;
    }

    // Advance until a seat needs a decision or the game ends.
    async step(): Promise<StepResult> {
        while (!this._isDone()) {
            await this.engine.tick(this.game, 1);
            this.recorder?.capture(this.game);
            this._updateMemories();

            const deciding = [...this.seats.values()].filter(
                (s) => !s.player.defeated && this._needsDecision(s),
            );

            if (deciding.length) {
                this._saveRecording();
                return this._result(deciding.map((s) => s.index));
            }
        }

        this._saveRecording();
        return this._result([]);
    }

    // Once the game is over, save its recording, if it is being recorded.
    _saveRecording() {
        if (!this.recorder || !this._isDone()) return;

        this.recorder.save(this.recordFile + ".tmp");
        fs.renameSync(this.recordFile + ".tmp", this.recordFile!);
        this.recorder = null;
    }

    _needsDecision(seat: Seat): boolean {
        const tick = this.game.state.tick;
        const cycleStart =
            tick % this.game.settings.galaxy.productionTicks === 1;
        const defendKeys = this._proposeOrders(seat)
            .filter((o) => o.kind === "defend")
            .map((o) => o.key)
            .sort()
            .join(",");
        const newAttack = defendKeys
            .split(",")
            .some((k) => k && !seat.defendKeys.split(",").includes(k));

        seat.defendKeys = defendKeys;

        return (
            cycleStart ||
            newAttack ||
            tick - seat.lastDecisionTick >= this.maxGap
        );
    }

    _seat(index: number): Seat {
        const seat = this.seats.get(index);
        if (!seat) throw new Error(`Seat ${index} is not played by an agent.`);
        return seat;
    }

    // Standings: every seat's stars and ships, and the winner once there is
    // one.
    standings() {
        const game = this.game;
        const winner = game.state.winner?.toString();

        return {
            tick: game.state.tick,
            done: this._isDone(),
            winner: winner
                ? game.galaxy.players.findIndex(
                      (p) => p._id.toString() === winner,
                  )
                : null,
            players: game.galaxy.players.map((p, index) => {
                const id = p._id.toString();
                const stars = game.galaxy.stars.filter(
                    (s) => s.ownedByPlayerId?.toString() === id,
                );
                const carriers = game.galaxy.carriers.filter(
                    (c) => c.ownedByPlayerId?.toString() === id,
                );

                return {
                    seat: index,
                    alias: p.alias,
                    agent: this.seats.has(index),
                    hired: this.seats.get(index)?.hired ?? 0,
                    hiresRefused: this.seats.get(index)?.hiresRefused ?? 0,
                    gifted: this.seats.get(index)?.gifted ?? 0,
                    alliedTick: this.seats.get(index)?.alliedTick ?? null,
                    allies: game.galaxy.players.filter(
                        (q) =>
                            q !== p &&
                            this.engine.container.diplomacyService.getDiplomaticStatusToPlayer(
                                game,
                                p._id,
                                q._id,
                            ).actualStatus === "allies",
                    ).length,
                    stars: stars.length,
                    ships:
                        stars.reduce((a, s) => a + (s.ships ?? 0), 0) +
                        carriers.reduce((a, c) => a + (c.ships ?? 0), 0),
                    defeated: p.defeated,
                };
            }),
        };
    }
}
