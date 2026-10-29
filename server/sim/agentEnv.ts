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
    // Built-in AI tuning for this seat. Fields left out keep their defaults.
    config?: Partial<AiConfig>;
    // Technology to research now and next.
    research?: string;
    // Under fog: what the agent believes, for the executor to act on. ships
    // gives believed ships per star index, used where the seat cannot see
    // a star's ships. Stars left out (or null) show the last ships seen.
    beliefs?: { ships?: (number | null)[] };
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
    defendKeys: string;
    memory: SeatMemory;
}

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

                const restore = this._fog(seat, true);
                try {
                    await ai.playForAgent(eventService, game, seat.player);
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
                defendKeys: "",
                memory: new SeatMemory(this.game.galaxy.stars.length),
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
        };
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

    act(index: number, decision: AgentDecision) {
        const seat = this._seat(index);
        seat.decision = decision;
        seat.lastDecisionTick = this.game.state.tick;

        this.engine.setAiConfig(
            this.game,
            seat.player._id,
            decision.config ?? {},
        );

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
