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
import mongoose from "mongoose";
import { SimEngine } from "./engine";
import { Game } from "../services/types/Game";
import { Player } from "../services/types/Player";
import { AiConfig } from "../services/types/Ai";
import { AiOrderCandidate } from "../services/ai";

export interface AgentDecision {
    // Score per order key; orders left out are dropped. Omit to keep the
    // built-in AI's choice of orders and scores.
    orders?: Record<string, number>;
    // Built-in AI tuning for this seat. Fields left out keep their defaults.
    config?: Partial<AiConfig>;
    // Technology to research now and next.
    research?: string;
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
}

export interface Observation {
    seat: number;
    tick: number;
    cycle: number;
    // Ticks until the next production (end of cycle).
    ticksToProduction: number;
    defeated: boolean;
    candidates: AiOrderCandidate[];
    // The seat's fog-of-war view, as GET /api/game/:gameId/galaxy returns
    // it. Only when asked for, since it is slower to build.
    view?: any;
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
}

export class AgentEnv {
    engine = new SimEngine();
    game!: Game;
    seats = new Map<number, Seat>();
    maxTicks = 3000;
    maxGap = 4;

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
                await ai.playForAgent(eventService, game, seat.player);
            }
        };

        ai.getOrderScores = (game, player) => {
            const seat = this._seatOf(player);
            const orders = seat?.decision.orders;
            return orders ? new Map(Object.entries(orders)) : null;
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
            });
        }

        return this._result([...this.seats.keys()]);
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

    async observe(index: number, options: { view?: boolean } = {}) {
        const seat = this._seat(index);
        const ticks = this.game.settings.galaxy.productionTicks;
        const tick = this.game.state.tick;

        const observation: Observation = {
            seat: index,
            tick,
            cycle: Math.floor(tick / ticks),
            ticksToProduction: ticks - (tick % ticks),
            defeated: seat.player.defeated,
            candidates: this.engine.container.aiService.proposeOrders(
                this.game,
                seat.player,
            ),
        };

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

            const deciding = [...this.seats.values()].filter(
                (s) => !s.player.defeated && this._needsDecision(s),
            );

            if (deciding.length) {
                return this._result(deciding.map((s) => s.index));
            }
        }

        return this._result([]);
    }

    _needsDecision(seat: Seat): boolean {
        const tick = this.game.state.tick;
        const cycleStart =
            tick % this.game.settings.galaxy.productionTicks === 1;
        const defendKeys = this.engine.container.aiService
            .proposeOrders(this.game, seat.player)
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
