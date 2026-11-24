import "../../env";
import { SeatMemory, seatFrame } from "../../fog";
import { VISIBLE, HIDDEN } from "../../beliefData";
import { BeliefWorld } from "../admiralEnv";
import { AdmiralDecision } from "../tacticalExecutor";
import {
    DiplomaticPolicy,
    buildDiplomaticPolicy,
    buildPeaceCancellationActions,
} from "./diplomaticPolicy";
import { parseNotesDirectives } from "./notesParser";
import {
    computeResearchMasks,
    coordinateResearch,
    detectReceivedTechnologies,
} from "./researchCoordinator";
import {
    estimateScheduledActionsReserve,
    recordAiCommittedCarriers,
    updateCarrierPins,
} from "./sharedSeat";
import { computeLiveOwnFacts, hydrateShadowFromGalaxy } from "./shadowHydrator";
import {
    createShadowEngine,
    dispatchLiveActions,
    recordShadowTickActions,
} from "./commandRecorder";
import {
    LiveStateStore,
    deserializeSeatMemory,
    serializeSeatMemory,
} from "./stateStore";
import { SolarisLiveClient } from "./solarisClient";
import { PythonAdmiralBridge } from "./pythonRpc";
import {
    LiveAction,
    LiveAlert,
    LiveDaemonConfig,
    OperatingMode,
    PersistedGameState,
} from "./types";

export interface CycleExecutionReport {
    tick: number;
    mode: OperatingMode;
    trigger: string;
    plannedActions: LiveAction[];
    dispatchedCount: number;
    abortedOnConflict: boolean;
    replanTriggered: boolean;
    alerts: LiveAlert[];
    diplomacy: DiplomaticPolicy;
}

/**
 * Orchestrates the Live Human-in-the-Loop Diplomatic `Admiral` Daemon:
 * - Dedicated multi-client login + Socket.IO real-time listeners
 * - Per-game SQLite/JSON persistence + Time-Machine mid-game attach backfill
 * - 3-Tier GUI diplomacy + private `attack:` covert override + 5 safety invariants
 * - Instant mid-tick tech reception handling + human GUI research pins + cash reserve
 * - Human-pinned carrier protection + coalesced sequential REST dispatch
 */
export class LiveAdmiralDaemon {
    readonly config: LiveDaemonConfig;
    readonly client: SolarisLiveClient;
    readonly store: LiveStateStore;
    readonly python: PythonAdmiralBridge;
    private readonly shadow = createShadowEngine();

    private persisted!: PersistedGameState;
    private seatMemory: SeatMemory | null = null;
    private myPlayerId: string | null = null;
    private mySeatIndex = -1;
    private pythonInitialized = false;
    private running = false;
    private activeCycleLock: Promise<any> = Promise.resolve();
    private pendingSocketTechs: {
        technology: string;
        level: number;
        fromPlayerId?: string;
    }[] = [];
    private monthlyBannedStarSpecs: number[] = [];
    private monthlyBannedCarrierSpecs: number[] = [];

    constructor(config: LiveDaemonConfig) {
        this.config = config;
        this.client = new SolarisLiveClient({
            apiUrl: config.apiUrl,
            email: config.email,
            password: config.password,
            cookie: config.cookie,
            maxRequestsPerSecond: config.maxRequestsPerSecond,
        });
        this.store = new LiveStateStore(config.statePath);
        this.python = new PythonAdmiralBridge();
    }

    getPersistedState(): PersistedGameState {
        return this.persisted;
    }

    getPlayerId(): string | null {
        return this.myPlayerId;
    }

    private emitAlert(alert: LiveAlert) {
        if (this.persisted) {
            this.persisted.human.alerts.push(alert);
        }
        this.config.onAlert?.(alert);
    }

    private serializeCycle<T>(fn: () => Promise<T>): Promise<T> {
        const next = this.activeCycleLock.then(fn, fn);
        this.activeCycleLock = next.catch(() => {});
        return next;
    }

    /**
     * Authenticates, discovers our player seat in `gameId`, loads persisted state,
     * optionally warms up `SeatMemory` & `BeliefTracker` via Time Machine, and
     * subscribes to real-time Socket.IO events.
     */
    async initialize(): Promise<{
        playerId: string;
        alias: string;
        tick: number;
    }> {
        await this.client.login();

        try {
            const bans = await this.client.getMonthlySpecialistBans();
            this.monthlyBannedStarSpecs = (bans?.star ?? []).map((s) => s.id);
            this.monthlyBannedCarrierSpecs = (bans?.carrier ?? []).map(
                (s) => s.id,
            );
        } catch {
            this.monthlyBannedStarSpecs = [];
            this.monthlyBannedCarrierSpecs = [];
        }

        const galaxy = await this.client.getGalaxy(this.config.gameId);
        const players: any[] = galaxy?.galaxy?.players ?? [];
        const meIdx = players.findIndex(
            (p) => p.hasPerspective === true || p.userId != null,
        );
        if (meIdx < 0) {
            throw new Error(
                `Authenticated user is not seated as a player in game ${this.config.gameId}.`,
            );
        }
        const me = players[meIdx];
        const playerId = String(me._id);
        this.myPlayerId = playerId;
        this.mySeatIndex = meIdx;

        const S = galaxy.galaxy.stars.length;
        this.persisted = this.store.load(this.config.gameId, playerId);
        this.seatMemory = deserializeSeatMemory(this.persisted.seatMemory, S);

        const hydrated = hydrateShadowFromGalaxy({
            engine: this.shadow.engine,
            galaxyPayload: galaxy,
            seatMemory: this.seatMemory,
            monthlyBannedStarSpecs: this.monthlyBannedStarSpecs,
            monthlyBannedCarrierSpecs: this.monthlyBannedCarrierSpecs,
        });

        await this.python.init({
            header: hydrated.staticHeader,
            seat: this.mySeatIndex,
            checkpoint: this.config.checkpointPath,
            trackerState: this.persisted.beliefTracker,
        });
        this.pythonInitialized = true;

        // Mid-game attach Time-Machine backfill (`GET /api/game/:gameId/galaxy?tick=T`):
        await this.backfillFromTimeMachineIfNeeded(galaxy);

        // Wire real-time Socket.IO callbacks:
        this.client.setCallbacks({
            onAuthExpired: () => {
                this.emitAlert({
                    timestamp: new Date().toISOString(),
                    tick: this.persisted?.lastProcessedTick ?? 0,
                    severity: "urgent",
                    code: "AUTH_EXPIRED",
                    message:
                        "Solaris session returned 401 Unauthorized — re-authenticate daemon.",
                });
            },
            onDiplomaticStatusChanged: () => {
                void this.handleRealtimeDiplomacyChange();
            },
            onTechnologyReceived: (payload) => {
                const tech = payload?.data?.technology;
                if (tech?.name) {
                    this.pendingSocketTechs.push({
                        technology: tech.name,
                        level: Number(tech.level ?? 0),
                        fromPlayerId: payload?.data?.fromPlayerId,
                    });
                }
                void this.handleRealtimeTechReceived();
            },
            onCreditsReceived: (payload) => {
                this.emitAlert({
                    timestamp: new Date().toISOString(),
                    tick: this.persisted?.lastProcessedTick ?? 0,
                    severity: "info",
                    code: "CREDITS_RECEIVED",
                    message: `Received ${payload?.data?.credits ?? 0} credits from player ${payload?.data?.fromPlayerId}.`,
                    details: payload?.data,
                });
            },
            onCreditsSpecialistsReceived: (payload) => {
                this.emitAlert({
                    timestamp: new Date().toISOString(),
                    tick: this.persisted?.lastProcessedTick ?? 0,
                    severity: "info",
                    code: "CREDITS_RECEIVED",
                    message: `Received ${payload?.data?.creditsSpecialists ?? 0} specialist token(s) from player ${payload?.data?.fromPlayerId}.`,
                    details: payload?.data,
                });
            },
            onGameMessageSent: (payload) => {
                this.emitAlert({
                    timestamp: new Date().toISOString(),
                    tick: this.persisted?.lastProcessedTick ?? 0,
                    severity: "info",
                    code: "CHAT_MESSAGE_NOTIFICATION",
                    message:
                        "Unread in-game chat activity received (handled strictly by human in GUI).",
                    details: {
                        conversationId: payload?.conversationId,
                    },
                });
            },
        });

        if (this.config.enableSockets !== false) {
            await this.client.connectSocket(this.config.gameId, playerId);
        }

        return {
            playerId,
            alias: me.alias,
            tick: Number(galaxy.state.tick ?? 0),
        };
    }

    private async backfillFromTimeMachineIfNeeded(currentGalaxy: any) {
        const curTick = Number(currentGalaxy?.state?.tick ?? 0);
        const timeMachineEnabled =
            currentGalaxy?.settings?.general?.timeMachine === "enabled";
        const maxBackfill = Math.max(0, this.config.backfillMaxTicks ?? 24);

        if (
            !timeMachineEnabled ||
            maxBackfill <= 0 ||
            curTick <= 1 ||
            this.persisted.lastProcessedTick >= 0 ||
            this.persisted.backfilledUpToTick >= curTick - 1
        ) {
            return;
        }

        const startTick = Math.max(1, curTick - maxBackfill);
        const frames: any[] = [];
        const tempMem = deserializeSeatMemory(
            this.persisted.seatMemory,
            currentGalaxy.galaxy.stars.length,
        );

        for (let t = startTick; t < curTick; t++) {
            try {
                const snap = await this.client.getGalaxy(this.config.gameId, t);
                if (!snap?.galaxy?.stars?.length) continue;
                const hyd = hydrateShadowFromGalaxy({
                    engine: this.shadow.engine,
                    galaxyPayload: snap,
                    seatMemory: tempMem,
                    monthlyBannedStarSpecs: this.monthlyBannedStarSpecs,
                    monthlyBannedCarrierSpecs: this.monthlyBannedCarrierSpecs,
                });
                frames.push(hyd.observation.frame);
            } catch {
                // Skip unavailable historical tick snapshots gracefully.
            }
        }

        if (frames.length > 0) {
            this.seatMemory = tempMem;
            await this.python.ingestFrames(frames);
        }
        this.persisted.backfilledUpToTick = curTick - 1;
        this.persisted.seatMemory = serializeSeatMemory(this.seatMemory!);
        this.store.save(this.persisted);
    }

    /**
     * Immediate mid-tick reaction to `playerDiplomaticStatusChanged`:
     * cancels delayed orbital attacks & truncates/alerts on in-flight carriers
     * headed toward newly `ALLIED` or `PROTECTED` players without waiting for
     * the next tick.
     */
    async handleRealtimeDiplomacyChange(): Promise<LiveAction[]> {
        return this.serializeCycle(async () => {
            if (!this.myPlayerId || !this.seatMemory) return [];
            const [galaxy, dipStatuses, notesResp] = await Promise.all([
                this.client.getGalaxy(this.config.gameId),
                this.client.getDiplomacy(this.config.gameId).catch(() => []),
                this.client
                    .getNotes(this.config.gameId)
                    .catch(() => ({ notes: undefined })),
            ]);
            const tick = Number(galaxy.state.tick ?? 0);
            const notesRes = parseNotesDirectives(
                notesResp?.notes,
                this.persisted.human.lastValidDirectives,
            );
            if (notesRes.ok && notesRes.hasBlock) {
                this.persisted.human.lastValidDirectives = notesRes.directives;
            }
            const attackList = notesRes.hasBlock
                ? notesRes.directives.attack
                : (this.config.attackOverrides ?? []);
            const neutralBorder =
                notesRes.directives.neutralBorder ??
                this.config.neutralBorderPosture;
            const effectiveMode = notesRes.directives.mode ?? this.config.mode;

            const diplomacy = buildDiplomaticPolicy({
                myPlayerId: this.myPlayerId,
                players: galaxy.galaxy.players,
                diplomaticStatuses: dipStatuses,
                attackList,
                neutralBorderPosture: neutralBorder,
                tick,
                onAlert: (a) => this.emitAlert(a),
            });

            const hyd = hydrateShadowFromGalaxy({
                engine: this.shadow.engine,
                galaxyPayload: galaxy,
                diplomaticStatuses: dipStatuses,
                seatMemory: this.seatMemory.copy(),
                diplomacy,
            });
            const ownCarriers = hyd.shadowGame.galaxy.carriers.filter(
                (c) => c.ownedByPlayerId?.toString() === this.myPlayerId,
            );
            const pinnedIds = updateCarrierPins({
                ownCarriers,
                human: this.persisted.human,
                tick,
                lastProcessedTick: tick,
                graceTicks: this.config.carrierGraceTicks,
            });

            const cancelActions = buildPeaceCancellationActions({
                game: hyd.shadowGame,
                policy: diplomacy,
                pinnedCarrierIds: pinnedIds,
                tick,
                calculateEta: (car, wp) =>
                    this.shadow.engine.container.waypointService.calculateWaypointTicksEta(
                        hyd.shadowGame,
                        car,
                        wp,
                    ) || 1,
                onAlert: (a) => this.emitAlert(a),
                mutateShadow: true,
            });

            if (
                cancelActions.length > 0 &&
                effectiveMode === "autonomous" &&
                !galaxy.state.locked
            ) {
                const dispatchedSet = new Set(
                    this.persisted.human.dispatchedIdempotencyKeys,
                );
                const res = await dispatchLiveActions({
                    client: this.client,
                    gameId: this.config.gameId,
                    tick,
                    actions: cancelActions,
                    batchScope: `dip-${Date.now()}`,
                    nextTickDate: galaxy.state.nextTickDate,
                    maxRequestsPerSecond: this.config.maxRequestsPerSecond,
                    deadlineSafetyMs: 0,
                    dispatchedKeys: dispatchedSet,
                    onKeyDispatched: (k) =>
                        this.persisted.human.dispatchedIdempotencyKeys.push(k),
                    onAlert: (a) => this.emitAlert(a),
                });
                if (!res.abortedOnConflict) {
                    recordAiCommittedCarriers({
                        postShadowCarriers: hyd.shadowGame.galaxy.carriers,
                        myPlayerId: this.myPlayerId,
                        pinnedCarrierIds: pinnedIds,
                        remappedCarrierIds: res.remappedCarrierIds,
                        human: this.persisted.human,
                        tick,
                    });
                }
            }

            this.store.save(this.persisted);
            return cancelActions;
        });
    }

    /**
     * Immediate mid-tick reaction to `playerTechnologyReceived`:
     * re-evaluates `researchingNow` / `researchingNext` within seconds of an
     * ally sending us a technology so science is never wasted on a wiped tech.
     */
    async handleRealtimeTechReceived(): Promise<LiveAction[]> {
        return this.serializeCycle(async () => {
            if (!this.myPlayerId || !this.seatMemory) return [];
            const [galaxy, dipStatuses, tradeEvents, notesResp] =
                await Promise.all([
                    this.client.getGalaxy(this.config.gameId),
                    this.client
                        .getDiplomacy(this.config.gameId)
                        .catch(() => []),
                    this.client
                        .getEvents(this.config.gameId, 0, 30, "trade")
                        .catch(() => ({ count: 0, events: [] })),
                    this.client
                        .getNotes(this.config.gameId)
                        .catch(() => ({ notes: undefined })),
                ]);

            const tick = Number(galaxy.state.tick ?? 0);
            const notesRes = parseNotesDirectives(
                notesResp?.notes,
                this.persisted.human.lastValidDirectives,
            );
            if (notesRes.ok && notesRes.hasBlock) {
                this.persisted.human.lastValidDirectives = notesRes.directives;
            }
            const effectiveMode = notesRes.directives.mode ?? this.config.mode;
            const expectTech = notesRes.hasBlock
                ? notesRes.directives.expectTech
                : (this.config.expectTech ?? []);
            const attackList = notesRes.hasBlock
                ? notesRes.directives.attack
                : (this.config.attackOverrides ?? []);

            const diplomacy = buildDiplomaticPolicy({
                myPlayerId: this.myPlayerId,
                players: galaxy.galaxy.players,
                diplomaticStatuses: dipStatuses,
                attackList,
                neutralBorderPosture:
                    notesRes.directives.neutralBorder ??
                    this.config.neutralBorderPosture,
                tick,
                onAlert: (a) => this.emitAlert(a),
            });

            const hyd = hydrateShadowFromGalaxy({
                engine: this.shadow.engine,
                galaxyPayload: galaxy,
                diplomaticStatuses: dipStatuses,
                seatMemory: this.seatMemory.copy(),
                diplomacy,
                monthlyBannedStarSpecs: this.monthlyBannedStarSpecs,
                monthlyBannedCarrierSpecs: this.monthlyBannedCarrierSpecs,
            });

            const socketTechs = this.pendingSocketTechs.splice(0);
            const received = detectReceivedTechnologies({
                me: hyd.myPlayer,
                technologies: hyd.technologies,
                human: this.persisted.human,
                socketTechEvents: socketTechs,
                tradeEvents: tradeEvents.events,
            });
            if (received.length === 0) return [];

            const maskPlan = computeResearchMasks({
                engine: this.shadow.engine,
                game: hyd.shadowGame,
                me: hyd.myPlayer,
                technologies: hyd.technologies,
                diplomacy,
                expectTech,
            });

            const pyRes = await this.python.decide({
                obs: hyd.observation,
                diplomaticTiers: diplomacy.tiersBySeatIndex,
                neutralBorderPosture: diplomacy.neutralBorderPosture,
                researchAllowedMask: maskPlan.allowedMask,
                bannedStarSpecialists: this.monthlyBannedStarSpecs,
            });

            const resCoord = coordinateResearch({
                engine: this.shadow.engine,
                game: hyd.shadowGame,
                me: hyd.myPlayer,
                technologies: hyd.technologies,
                human: this.persisted.human,
                maskPlan,
                receivedTechs: received,
                resLogits: pyRes.resLogits,
                policyChoice: pyRes.decision.research ?? null,
                expectTech,
                isCycleStart: false,
                forceReevaluate: true,
                onAlert: (a) => this.emitAlert(a),
            });

            if (
                resCoord.actions.length > 0 &&
                effectiveMode === "autonomous" &&
                !galaxy.state.locked
            ) {
                const dispatchedSet = new Set(
                    this.persisted.human.dispatchedIdempotencyKeys,
                );
                await dispatchLiveActions({
                    client: this.client,
                    gameId: this.config.gameId,
                    tick,
                    actions: resCoord.actions,
                    batchScope: `tech-${Date.now()}`,
                    nextTickDate: galaxy.state.nextTickDate,
                    maxRequestsPerSecond: this.config.maxRequestsPerSecond,
                    deadlineSafetyMs: 0,
                    dispatchedKeys: dispatchedSet,
                    onKeyDispatched: (k) =>
                        this.persisted.human.dispatchedIdempotencyKeys.push(k),
                    onAlert: (a) => this.emitAlert(a),
                });
                this.persisted.human.lastAiResearch = {
                    now: resCoord.desiredNow,
                    next: resCoord.desiredNext,
                    tick,
                };
            }

            this.store.save(this.persisted);
            return resCoord.actions;
        });
    }

    /**
     * Executes one full synchronization + planning (+ optional live dispatch)
     * pass for the current tick.
     */
    async runOnce(trigger = "tick"): Promise<CycleExecutionReport> {
        if (!this.pythonInitialized || !this.myPlayerId || !this.seatMemory) {
            await this.initialize();
        }
        return this.serializeCycle(() =>
            this._executeCycleInternal(trigger, false),
        );
    }

    private async _executeCycleInternal(
        trigger: string,
        isConflictRetry: boolean,
    ): Promise<CycleExecutionReport> {
        const [galaxy, dipStatuses, tradeEvents, allEvents, notesResp] =
            await Promise.all([
                this.client.getGalaxy(this.config.gameId),
                this.client.getDiplomacy(this.config.gameId).catch(() => []),
                this.client
                    .getEvents(this.config.gameId, 0, 30, "trade")
                    .catch(() => ({ count: 0, events: [] })),
                this.client
                    .getEvents(this.config.gameId, 0, 30, "all")
                    .catch(() => ({ count: 0, events: [] })),
                this.client
                    .getNotes(this.config.gameId)
                    .catch(() => ({ notes: undefined })),
            ]);

        const tick = Number(galaxy.state.tick ?? 0);
        const cycleAlerts: LiveAlert[] = [];
        const recordAlert = (a: LiveAlert) => {
            cycleAlerts.push(a);
            this.emitAlert(a);
        };

        // 1. Parse `# === AI ===` private In-Game Notes directives (if present).
        const notesRaw = notesResp?.notes ?? null;
        const notesRes = parseNotesDirectives(
            notesRaw,
            this.persisted.human.lastValidDirectives,
        );
        for (const w of notesRes.warnings) {
            recordAlert({
                timestamp: new Date().toISOString(),
                tick,
                severity: "warn",
                code: "NOTES_PARSE_WARNING",
                message: w,
            });
        }
        if (notesRes.ok) {
            this.persisted.human.lastValidDirectives = notesRes.hasBlock
                ? notesRes.directives
                : null;
        }
        this.persisted.human.lastNotesRaw = notesRaw;

        const directives = notesRes.directives;
        const effectiveMode: OperatingMode =
            directives.mode ?? this.config.mode;
        const cashReserve =
            directives.cashReserve ?? this.config.cashReserveCredits;
        const neutralBorder =
            directives.neutralBorder ?? this.config.neutralBorderPosture;
        const attackList = notesRes.hasBlock
            ? directives.attack
            : (this.config.attackOverrides ?? []);
        const expectTech = notesRes.hasBlock
            ? directives.expectTech
            : (this.config.expectTech ?? []);

        // 2. Build 3-tier Diplomatic Policy.
        const diplomacy = buildDiplomaticPolicy({
            myPlayerId: this.myPlayerId!,
            players: galaxy.galaxy.players,
            diplomaticStatuses: dipStatuses,
            attackList,
            neutralBorderPosture: neutralBorder,
            tick,
            onAlert: recordAlert,
        });
        for (const [pid, entry] of diplomacy.entriesByPlayerId.entries()) {
            this.persisted.human.lastDiplomaticTiers[pid] = entry.tier;
        }

        // 3. Hydrate Shadow Game + SeatMemory + Fog Observation.
        const hyd = hydrateShadowFromGalaxy({
            engine: this.shadow.engine,
            galaxyPayload: galaxy,
            diplomaticStatuses: dipStatuses,
            seatMemory: this.seatMemory!,
            diplomacy,
            monthlyBannedStarSpecs: this.monthlyBannedStarSpecs,
            monthlyBannedCarrierSpecs: this.monthlyBannedCarrierSpecs,
        });

        // 4. Shared-seat carrier pin detection (`humanPinnedCarriers`).
        const ownCarriers = hyd.shadowGame.galaxy.carriers.filter(
            (c) => c.ownedByPlayerId?.toString() === this.myPlayerId,
        );
        const pinnedCarrierIds = updateCarrierPins({
            ownCarriers,
            human: this.persisted.human,
            tick,
            lastProcessedTick: this.persisted.lastProcessedTick,
            graceTicks: this.config.carrierGraceTicks,
        });

        // If `mode: paused`, persist updated SeatMemory/tech levels and stop cleanly!
        if (effectiveMode === "paused" || hyd.myPlayer.defeated) {
            for (const t of hyd.technologies) {
                this.persisted.human.lastKnownTechLevels[t] = Number(
                    hyd.myPlayer.research?.[t]?.level ?? 1,
                );
            }
            this.persisted.lastProcessedTick = tick;
            this.persisted.seatMemory = serializeSeatMemory(this.seatMemory!);
            this.store.save(this.persisted);
            return {
                tick,
                mode: effectiveMode,
                trigger,
                plannedActions: [],
                dispatchedCount: 0,
                abortedOnConflict: false,
                replanTriggered: false,
                alerts: cycleAlerts,
                diplomacy,
            };
        }

        // 5. Detect received technologies & compute ResearchCoordinator masks.
        const socketTechs = this.pendingSocketTechs.splice(0);
        const receivedTechs = detectReceivedTechnologies({
            me: hyd.myPlayer,
            technologies: hyd.technologies,
            human: this.persisted.human,
            socketTechEvents: socketTechs,
            tradeEvents: tradeEvents.events,
            allRecentEvents: allEvents.events,
        });
        const maskPlan = computeResearchMasks({
            engine: this.shadow.engine,
            game: hyd.shadowGame,
            me: hyd.myPlayer,
            technologies: hyd.technologies,
            diplomacy,
            expectTech,
        });

        // Identify contested unclaimed stars for Invariant 1 & Invariant 5 masking.
        const starIndexById = new Map<string, number>();
        hyd.shadowGame.galaxy.stars.forEach((s, idx) =>
            starIndexById.set(s._id.toString(), idx),
        );
        const contestedUnownedStars: number[] = [];
        for (const c of hyd.shadowGame.galaxy.carriers) {
            if (
                c.orbiting != null ||
                !c.waypoints?.length ||
                !c.ownedByPlayerId
            )
                continue;
            const ownerStr = c.ownedByPlayerId.toString();
            if (ownerStr === this.myPlayerId) continue;
            if (
                diplomacy.isAllied(ownerStr) ||
                diplomacy.isProtected(ownerStr)
            ) {
                const dIdx = starIndexById.get(
                    c.waypoints[0].destination.toString(),
                );
                if (
                    dIdx !== undefined &&
                    !hyd.shadowGame.galaxy.stars[dIdx].ownedByPlayerId
                ) {
                    contestedUnownedStars.push(dIdx);
                }
            }
        }

        const prodTicks = hyd.shadowGame.settings.galaxy.productionTicks || 20;
        const isCycleStart = tick % prodTicks === 1;
        const allBannedStarSpecs = [
            ...(hyd.shadowGame.settings.specialGalaxy.specialistBans?.star ??
                []),
            ...this.monthlyBannedStarSpecs,
        ];

        // 6. Query Python `AdmiralNet` + `BeliefTracker` (with optional 1-cycle lookahead).
        let chosenDecision: AdmiralDecision;
        let resLogits: number[];

        if (this.config.useSearchPlanner && isCycleStart) {
            const prop = await this.python.proposeSearch({
                obs: hyd.observation,
                diplomaticTiers: diplomacy.tiersBySeatIndex,
                neutralBorderPosture: neutralBorder,
                researchAllowedMask: maskPlan.allowedMask,
                bannedStarSpecialists: allBannedStarSpecs,
                contestedUnownedStars,
                nPlans: this.config.searchPlans ?? 4,
                nWorlds: this.config.searchWorlds ?? 2,
                seed: tick + 1,
            });
            resLogits = prop.resLogits;
            this.persisted.beliefTracker = prop.trackerState;
            chosenDecision = await this._selectBestPlanViaLookahead({
                hyd,
                plans: prop.plans,
                worlds: prop.worlds,
                diplomacy,
                pinnedCarrierIds,
                cashReserve,
            });
        } else {
            const pyOut = await this.python.decide({
                obs: hyd.observation,
                diplomaticTiers: diplomacy.tiersBySeatIndex,
                neutralBorderPosture: neutralBorder,
                researchAllowedMask: maskPlan.allowedMask,
                bannedStarSpecialists: allBannedStarSpecs,
                contestedUnownedStars,
            });
            chosenDecision = pyOut.decision;
            resLogits = pyOut.resLogits;
            this.persisted.beliefTracker = pyOut.trackerState;
        }

        // 7. Coordinate Research (`SetResearchNow` / `SetResearchNext`).
        const resCoord = coordinateResearch({
            engine: this.shadow.engine,
            game: hyd.shadowGame,
            me: hyd.myPlayer,
            technologies: hyd.technologies,
            human: this.persisted.human,
            maskPlan,
            receivedTechs,
            resLogits,
            policyChoice: chosenDecision.research ?? null,
            expectTech,
            isCycleStart,
            onAlert: recordAlert,
        });

        // 8. Compute scheduled bulk upgrade reserve & run Shadow-Sim CommandRecorder!
        const scheduledReserve = estimateScheduledActionsReserve(
            this.shadow.engine,
            hyd.shadowGame,
            hyd.myPlayer,
        );

        const recorded = await recordShadowTickActions({
            engine: this.shadow.engine,
            executor: this.shadow.executor,
            shadowGame: hyd.shadowGame,
            mySeatIndex: hyd.mySeatIndex,
            seatMemory: this.seatMemory!,
            view: hyd.view,
            decision: chosenDecision,
            researchActions: resCoord.actions,
            diplomacy,
            pinnedCarrierIds,
            cashReserveCredits: cashReserve,
            scheduledActionCredits: scheduledReserve,
            allowEmergencyDefenseReserveDip:
                this.config.allowEmergencyDefenseReserveDip,
            bannedStarSpecialists: new Set(allBannedStarSpecs),
        });
        for (const a of recorded.alerts) {
            recordAlert(a);
        }

        // 9. Handle Operating Mode (`advisor` | `approval` | `autonomous`).
        let shouldDispatch = false;
        if (effectiveMode === "autonomous") {
            shouldDispatch = true;
        } else if (
            effectiveMode === "approval" &&
            recorded.actions.length > 0
        ) {
            shouldDispatch = this.config.approvalHandler
                ? await this.config.approvalHandler(recorded.actions, tick)
                : false;
        }

        let dispatchedCount = 0;
        let abortedOnConflict = false;
        let remapped = new Map<string, string>();
        let conflictError: string | undefined;

        if (
            shouldDispatch &&
            recorded.actions.length > 0 &&
            !galaxy.state.locked
        ) {
            const dispatchedSet = new Set(
                this.persisted.human.dispatchedIdempotencyKeys,
            );
            const batchRes = await dispatchLiveActions({
                client: this.client,
                gameId: this.config.gameId,
                tick,
                actions: recorded.actions,
                batchScope: trigger,
                nextTickDate: galaxy.state.nextTickDate,
                maxRequestsPerSecond: this.config.maxRequestsPerSecond,
                deadlineSafetyMs: this.config.deadlineSafetyMs,
                dispatchedKeys: dispatchedSet,
                onKeyDispatched: (k) =>
                    this.persisted.human.dispatchedIdempotencyKeys.push(k),
                onAlert: recordAlert,
            });
            dispatchedCount = batchRes.dispatchedCount;
            abortedOnConflict = batchRes.abortedOnConflict;
            conflictError = batchRes.conflictError;
            remapped = batchRes.remappedCarrierIds;

            if (!abortedOnConflict) {
                this.persisted.human.lastAiResearch = {
                    now: resCoord.desiredNow,
                    next: resCoord.desiredNext,
                    tick,
                };
                const builtTempIds = new Set(
                    recorded.actions
                        .filter((a) => a.type === "BuildCarrier")
                        .map((a) => (a as any).tempCarrierId as string),
                );
                recordAiCommittedCarriers({
                    postShadowCarriers: recorded.postShadowGame.galaxy.carriers,
                    myPlayerId: this.myPlayerId!,
                    pinnedCarrierIds,
                    remappedCarrierIds: remapped,
                    unmappedTempCarrierIds: builtTempIds,
                    human: this.persisted.human,
                    tick,
                });
            }
        }

        this.persisted.lastProcessedTick = tick;
        this.persisted.seatMemory = serializeSeatMemory(this.seatMemory!);
        this.persisted.human.auditLog.push({
            timestamp: new Date().toISOString(),
            tick,
            mode: effectiveMode,
            trigger,
            plannedActions: recorded.actions,
            dispatchedCount,
            remappedCarriers: Object.fromEntries(remapped.entries()),
            conflictError,
        });
        this.store.save(this.persisted);

        // Stale-State Conflict Recovery (Section 2.4): re-fetch & re-plan at most once!
        if (abortedOnConflict && !isConflictRetry) {
            const retryReport = await this._executeCycleInternal(
                `${trigger}:conflict_replan`,
                true,
            );
            return {
                ...retryReport,
                replanTriggered: true,
                alerts: [...cycleAlerts, ...retryReport.alerts],
            };
        }

        return {
            tick,
            mode: effectiveMode,
            trigger,
            plannedActions: recorded.actions,
            dispatchedCount,
            abortedOnConflict,
            replanTriggered: false,
            alerts: cycleAlerts,
            diplomacy,
        };
    }

    private async _selectBestPlanViaLookahead(params: {
        hyd: ReturnType<typeof hydrateShadowFromGalaxy>;
        plans: AdmiralDecision[];
        worlds: BeliefWorld[];
        diplomacy: DiplomaticPolicy;
        pinnedCarrierIds: ReadonlySet<string>;
        cashReserve: number;
    }): Promise<AdmiralDecision> {
        const { hyd, plans, worlds, diplomacy, pinnedCarrierIds, cashReserve } =
            params;
        if (plans.length <= 1) return plans[0] ?? {};

        const engine = this.shadow.engine;
        const executor = this.shadow.executor;
        const baseGame = hyd.shadowGame;
        const mySeat = hyd.mySeatIndex;
        const myId = baseGame.galaxy.players[mySeat]._id.toString();
        const horizon = baseGame.settings.galaxy.productionTicks || 20;

        // Build determinized fog-legal world templates preserving diplomatic statuses!
        const worldJsonTemplates = worlds.map((w) => {
            const copy = engine.clone(baseGame);
            copy.galaxy.stars.forEach((s, i) => {
                if (s.ownedByPlayerId?.toString() === myId) return;
                if (hyd.view.stars[i] !== VISIBLE) {
                    const sh = Math.max(0, Math.round(w.ships[i] ?? 0));
                    s.ships = sh;
                    s.shipsActual = sh;
                }
                if (hyd.view.stars[i] === HIDDEN) {
                    s.infrastructure = {
                        economy: Math.max(0, Math.round(w.economy?.[i] ?? 0)),
                        industry: Math.max(0, Math.round(w.industry?.[i] ?? 0)),
                        science: Math.max(0, Math.round(w.science?.[i] ?? 0)),
                    };
                }
            });
            for (const p of copy.galaxy.players) {
                const pid = p._id.toString();
                if (pid === myId) continue;
                p.credits = p.defeated ? 0 : 100;
                p.creditsSpecialists = 0;
                // Keep ALLIED / PROTECTED opponents passive toward non-enemies in rollout
                // while HOSTILE opponents use built-in AI (`userId = null`).
                p.userId = diplomacy.isHostile(pid) ? null : p._id;
            }
            return engine.toJSON(copy);
        });

        const leaves: {
            frame: any;
            ownFacts: any;
            players: { stars: number; ships: number; defeated: boolean }[];
            ticksToEnd: number | null;
        }[] = [];

        const origRandom = Math.random;
        try {
            for (const plan of plans) {
                for (let wi = 0; wi < worldJsonTemplates.length; wi++) {
                    const simGame = engine.fromJSON(
                        JSON.parse(JSON.stringify(worldJsonTemplates[wi])),
                    );
                    const simPlayer = simGame.galaxy.players[mySeat];
                    const simMem = this.seatMemory!.copy();
                    simMem.update(engine, simGame, simPlayer);

                    engine.seed(101 + wi * 997 + baseGame.state.tick);
                    for (
                        let step = 0;
                        step < horizon && !simGame.state.endDate;
                        step++
                    ) {
                        await executor.executeTick(
                            simGame,
                            simPlayer,
                            plan,
                            undefined,
                            simMem.view?.stars ?? null,
                            {
                                diplomacy: diplomacy.toExecutorHooks(),
                                pinnedCarrierIds,
                                respectIgnoreBulkUpgrade: true,
                                cashReserveCredits: cashReserve,
                            },
                        );
                        await engine.tick(simGame, 1);
                        simMem.update(engine, simGame, simPlayer);
                    }

                    const lfFrame = seatFrame(
                        simGame,
                        simMem,
                        hyd.technologies,
                    );
                    const lfOwn = computeLiveOwnFacts({
                        engine,
                        game: simGame,
                        player: simPlayer,
                        technologies: hyd.technologies,
                        view: simMem.view!,
                        diplomacy,
                    });
                    const standings = simGame.galaxy.players.map((p, idx) => ({
                        stars: lfFrame.stats[idx][0],
                        ships: lfFrame.stats[idx][1],
                        defeated: Boolean(p.defeated),
                    }));
                    leaves.push({
                        frame: lfFrame,
                        ownFacts: lfOwn,
                        players: standings,
                        ticksToEnd:
                            simGame.state.ticksToEnd != null
                                ? Math.max(0, simGame.state.ticksToEnd)
                                : null,
                    });
                }
            }
        } finally {
            Math.random = origRandom;
        }

        const { scores } = await this.python.scoreLeaves({
            leaves,
            diplomaticTiers: diplomacy.tiersBySeatIndex,
            neutralBorderPosture: diplomacy.neutralBorderPosture,
        });

        const K = worldJsonTemplates.length;
        let bestPlanIdx = 0;
        let bestAvg = -Infinity;
        for (let m = 0; m < plans.length; m++) {
            let sum = 0;
            for (let wi = 0; wi < K; wi++) {
                sum += scores[m * K + wi] ?? 0;
            }
            const avg = sum / Math.max(1, K);
            if (avg > bestAvg) {
                bestAvg = avg;
                bestPlanIdx = m;
            }
        }

        return plans[bestPlanIdx];
    }

    /**
     * Starts the continuous real-time tick polling loop (sleeping until
     * `nextTickDate + tickOffsetMs`, then polling `/state` every `pollIntervalMs`).
     */
    async start(): Promise<void> {
        if (!this.pythonInitialized) {
            await this.initialize();
        }
        this.running = true;

        while (this.running) {
            try {
                const stateResp = await this.client.getState(
                    this.config.gameId,
                );
                const state = stateResp?.state ?? {};
                if (state.endDate || state.winner) {
                    break;
                }

                const curTick = Number(state.tick ?? 0);
                if (
                    !state.paused &&
                    !state.locked &&
                    state.startDate &&
                    curTick > this.persisted.lastProcessedTick
                ) {
                    await this.runOnce("tick");
                }

                // Compute sleep duration until `nextTickDate + tickOffsetMs`.
                let waitMs = this.config.pollIntervalMs;
                if (state.nextTickDate) {
                    const targetMs =
                        new Date(state.nextTickDate).getTime() +
                        this.config.tickOffsetMs;
                    const delta = targetMs - Date.now();
                    if (delta > this.config.pollIntervalMs) {
                        waitMs = Math.min(delta, 60_000);
                    }
                }
                await new Promise((r) => setTimeout(r, Math.max(250, waitMs)));
            } catch {
                await new Promise((r) =>
                    setTimeout(r, Math.max(1000, this.config.pollIntervalMs)),
                );
            }
        }
    }

    stop() {
        this.running = false;
        this.client.disconnectSocket();
        this.python.close();
        this.store.close();
    }
}
