import crypto from "crypto";
import { ValidationError } from "@solaris/common";
import { SimEngine } from "../../engine";
import { SeatMemory, applyFog } from "../../fog";
import { SeatView } from "../../beliefData";
import {
    AdmiralDecision,
    TacticalExecutor,
    TacticalExecutionOptions,
} from "../tacticalExecutor";
import { installFastCombat } from "../fastDefense";
import { Game } from "../../../services/types/Game";
import { InfrastructureType } from "../../../services/types/Star";
import {
    DiplomaticPolicy,
    buildPeaceCancellationActions,
} from "./diplomaticPolicy";
import { hashWaypoints, normalizeWaypoints } from "./sharedSeat";
import { SolarisHttpError, SolarisLiveClient } from "./solarisClient";
import { LiveAction, LiveAlert } from "./types";

export interface ShadowPlanOutput {
    actions: LiveAction[];
    postShadowGame: Game;
    alerts: LiveAlert[];
}

export interface DispatchBatchResult {
    dispatchedCount: number;
    skippedDuplicateCount: number;
    deferredByDeadline: boolean;
    abortedOnConflict: boolean;
    conflictError?: string;
    remappedCarrierIds: Map<string, string>;
}

/**
 * Prepares a shadow `SimEngine` whose in-memory upgrade/carrier services never
 * touch Mongoose repositories.
 */
export function createShadowEngine(): {
    engine: SimEngine;
    executor: TacticalExecutor;
} {
    const engine = new SimEngine();
    installFastCombat(engine.container.combatService);
    const starUpgrade = engine.container.starUpgradeService as any;
    starUpgrade.gameRepo = {
        bulkWrite: async () => {},
        updateOne: async () => {},
    };
    const executor = new TacticalExecutor(engine.container);
    return { engine, executor };
}

/**
 * Runs `TacticalExecutor.executeTick` inside fog on `shadowGame` while enforcing
 * all diplomatic, treasury, research, and shared-seat constraints, and records a
 * coalesced, dependency-ordered `LiveAction[]` list:
 *
 * 1. `SetResearchNow` / `SetResearchQueue`
 * 2. Pre-execution Invariant 2 peace/alliance waypoint cancellations
 * 3. `HireStarSpecialist` -> `UpgradeStar` / `BuildWarpGate`
 * 4. Coalesced star ship pool unloads (`TransferShips` where `after < before`)
 *    -> `BuildCarrier(starId, finalShips)` -> `TransferShips` (`after > before`)
 * 5. `HireCarrierSpecialist` (before `SaveWaypoints` so Explorer `+3` range validates!)
 * 6. `SaveWaypoints`
 */
export async function recordShadowTickActions(params: {
    engine: SimEngine;
    executor: TacticalExecutor;
    shadowGame: Game;
    mySeatIndex: number;
    seatMemory: SeatMemory;
    view: SeatView;
    decision: AdmiralDecision;
    researchActions: LiveAction[];
    diplomacy: DiplomaticPolicy;
    pinnedCarrierIds: ReadonlySet<string>;
    cashReserveCredits: number;
    scheduledActionCredits: number;
    allowEmergencyDefenseReserveDip?: boolean;
    bannedStarSpecialists?: ReadonlySet<number>;
}): Promise<ShadowPlanOutput> {
    const {
        engine,
        executor,
        shadowGame,
        mySeatIndex,
        seatMemory,
        view,
        diplomacy,
        pinnedCarrierIds,
    } = params;
    const player = shadowGame.galaxy.players[mySeatIndex];
    const myId = player._id.toString();
    const tick = shadowGame.state.tick;
    const alerts: LiveAlert[] = [];

    const emitAlert = (a: LiveAlert) => alerts.push(a);

    // Hard-disable autonomous diplomacy & mask `attackDesire` on ALLIED / PROTECTED stars.
    const safeDecision: AdmiralDecision = structuredClone(
        params.decision ?? {},
    );
    safeDecision.court = false;
    safeDecision.betray = false;

    const S = shadowGame.galaxy.stars.length;
    const maskedDesire = new Array<number>(S);
    for (let i = 0; i < S; i++) {
        const rawD = safeDecision.attackDesire?.[i] ?? 1.0;
        const ownerStr = shadowGame.galaxy.stars[i].ownedByPlayerId?.toString();
        if (ownerStr && ownerStr !== myId && !diplomacy.isHostile(ownerStr)) {
            maskedDesire[i] = 0;
        } else {
            maskedDesire[i] = rawD;
        }
    }
    safeDecision.attackDesire = maskedDesire;

    // Stage 0: Invariant 2 — cancel staged delayed attacks & truncate in-flight multi-hop
    // attacks on any player who is currently `ALLIED` or `PROTECTED`.
    const peaceActions = buildPeaceCancellationActions({
        game: shadowGame,
        policy: diplomacy,
        pinnedCarrierIds,
        tick,
        calculateEta: (car, wp) =>
            engine.container.waypointService.calculateWaypointTicksEta(
                shadowGame,
                car,
                wp,
            ) || 1,
        onAlert: emitAlert,
        mutateShadow: true,
    });

    // Apply research decisions onto shadow player so ETA/tech checks stay consistent.
    for (const rAct of params.researchActions) {
        if (rAct.type === "SetResearchNow") {
            player.researchingNow = rAct.preference as any;
        } else if (rAct.type === "SetResearchQueue") {
            player.researchQueue = [...rAct.queue] as any;
        }
    }

    // Snapshot pre-execution star & carrier state on our own stars/carriers.
    const beforeStars = new Map<
        string,
        {
            economy: number;
            industry: number;
            science: number;
            warpGate: boolean;
            ships: number;
            name: string;
        }
    >();
    for (const s of shadowGame.galaxy.stars) {
        if (s.ownedByPlayerId?.toString() !== myId) continue;
        beforeStars.set(s._id.toString(), {
            economy: Number(s.infrastructure?.economy ?? 0),
            industry: Number(s.infrastructure?.industry ?? 0),
            science: Number(s.infrastructure?.science ?? 0),
            warpGate: Boolean(s.warpGate),
            ships: Math.floor(s.shipsActual ?? s.ships ?? 0),
            name: s.name,
        });
    }

    const beforeCarriers = new Map<
        string,
        {
            ships: number;
            orbiting: string | null;
            waypointsHash: string;
            specialistId: number | null;
        }
    >();
    for (const c of shadowGame.galaxy.carriers) {
        if (c.ownedByPlayerId?.toString() !== myId) continue;
        const wps = normalizeWaypoints(c.waypoints);
        beforeCarriers.set(c._id.toString(), {
            ships: Number(c.ships ?? 0),
            orbiting: c.orbiting ? c.orbiting.toString() : null,
            waypointsHash: hashWaypoints(wps),
            specialistId:
                c.specialistId != null ? Number(c.specialistId) : null,
        });
    }

    // Stage 2a: Star Specialist hires (validated against bans & specialist budget).
    const starHireActions: LiveAction[] = [];
    for (const { star, specialist } of safeDecision.hire ?? []) {
        const target = shadowGame.galaxy.stars[star];
        if (!target || target.ownedByPlayerId?.toString() !== myId) continue;
        if (params.bannedStarSpecialists?.has(specialist)) continue;
        if (
            engine.container.specialistBanService.isStarSpecialistBanned(
                shadowGame,
                specialist,
            )
        ) {
            continue;
        }
        // If specialist currency is `credits`, guard `cashReserveCredits + scheduledActionCredits`!
        const specCurrency =
            shadowGame.settings.specialGalaxy.specialistsCurrency;
        if (specCurrency === "credits") {
            const specObj =
                engine.container.specialistService.getByIdStar(specialist);
            const cost = specObj
                ? engine.container.specialistService.getSpecialistActualCost(
                      shadowGame,
                      specObj,
                  ).credits
                : Infinity;
            const minKeep =
                params.cashReserveCredits + params.scheduledActionCredits;
            if ((player.credits ?? 0) - cost < minKeep) continue;
        }
        try {
            await engine.container.specialistHireService.hireStarSpecialist(
                shadowGame,
                player,
                target._id,
                specialist,
                engine.container.statisticsService,
            );
            starHireActions.push({
                type: "HireStarSpecialist",
                starId: target._id.toString(),
                starName: target.name,
                specialistId: specialist,
            });
        } catch (e: any) {
            if (!(e instanceof ValidationError)) throw e;
        }
    }

    // Run `TacticalExecutor.executeTick` inside `applyFog` with full live constraints!
    const believedShips = (idx: number) =>
        safeDecision.beliefs?.ships?.[idx] ?? undefined;
    const believedIndustry = safeDecision.beliefs?.industry
        ? (idx: number) => safeDecision.beliefs!.industry![idx] ?? undefined
        : undefined;

    const execOptions: TacticalExecutionOptions = {
        diplomacy: diplomacy.toExecutorHooks(),
        pinnedCarrierIds,
        respectIgnoreBulkUpgrade: true,
        cashReserveCredits: params.cashReserveCredits,
        scheduledActionCredits: params.scheduledActionCredits,
        allowEmergencyDefenseReserveDip:
            params.allowEmergencyDefenseReserveDip ?? true,
        onAlert: (ev) => {
            if (ev.kind === "surprise_attack") {
                emitAlert({
                    timestamp: new Date().toISOString(),
                    tick,
                    severity: "urgent",
                    code: "SURPRISE_ATTACK",
                    message: ev.message,
                    details: ev,
                });
            } else if (ev.kind === "emergency_reserve_dip") {
                emitAlert({
                    timestamp: new Date().toISOString(),
                    tick,
                    severity: "warn",
                    code: "EMERGENCY_RESERVE_DIP",
                    message: ev.message,
                    details: ev,
                });
            }
        },
    };

    // Deterministic temp ObjectId generator for newly built shadow carriers so
    // `tempCarrierId`s are distinct from live Mongo ObjectIds and stable in dry-runs.
    const restoreFog = applyFog(shadowGame, player, seatMemory, believedShips);
    try {
        await executor.executeTick(
            shadowGame,
            player,
            safeDecision,
            believedIndustry,
            view.stars,
            execOptions,
        );
    } finally {
        restoreFog();
    }

    // Stage 2b: Diff infrastructure upgrades & warp gates on own stars.
    const infraActions: LiveAction[] = [];
    const infraOrder: InfrastructureType[] = ["science", "industry", "economy"];
    for (const s of shadowGame.galaxy.stars) {
        const sid = s._id.toString();
        const before = beforeStars.get(sid);
        if (!before) continue;
        for (const infType of infraOrder) {
            const afterVal = Number(s.infrastructure?.[infType] ?? 0);
            const delta = afterVal - before[infType];
            for (let k = 0; k < delta; k++) {
                infraActions.push({
                    type: "UpgradeStar",
                    starId: sid,
                    starName: s.name,
                    infrastructure: infType,
                });
            }
        }
        if (!before.warpGate && Boolean(s.warpGate)) {
            infraActions.push({
                type: "BuildWarpGate",
                starId: sid,
                starName: s.name,
            });
        }
    }

    // Stage 3 & 4: Coalesce per-star ship unloads, `BuildCarrier(starId, finalShips)`,
    // and carrier ship loads so every intermediate HTTP step preserves exact ship sums!
    const unloadTransfers: LiveAction[] = [];
    const buildCarrierActions: LiveAction[] = [];
    const loadTransfers: LiveAction[] = [];
    const carrierHireActions: LiveAction[] = [];
    const saveWaypointActions: LiveAction[] = [];

    const afterCarriersByStar = new Map<string, any[]>();
    for (const c of shadowGame.galaxy.carriers) {
        if (c.ownedByPlayerId?.toString() !== myId) continue;
        const cid = c._id.toString();
        if (pinnedCarrierIds.has(cid)) continue;

        const before = beforeCarriers.get(cid);
        const orbitStarId =
            before?.orbiting ??
            (c.orbiting ? c.orbiting.toString() : null) ??
            (c.waypoints?.[0]?.source
                ? c.waypoints[0].source.toString()
                : null);

        if (orbitStarId && beforeStars.has(orbitStarId)) {
            let list = afterCarriersByStar.get(orbitStarId);
            if (!list) {
                list = [];
                afterCarriersByStar.set(orbitStarId, list);
            }
            list.push(c);
        }
    }

    for (const [starId, starBefore] of beforeStars.entries()) {
        const list = afterCarriersByStar.get(starId) ?? [];
        if (!list.length) continue;

        let runningStarShips = starBefore.ships;
        const existingUnload: any[] = [];
        const newlyBuilt: any[] = [];
        const existingLoad: any[] = [];

        for (const c of list) {
            const cid = c._id.toString();
            const before = beforeCarriers.get(cid);
            if (!before) {
                newlyBuilt.push(c);
            } else if (before.orbiting === starId) {
                const afterShips = Number(c.ships ?? 0);
                if (afterShips < before.ships) {
                    existingUnload.push(c);
                } else if (afterShips > before.ships) {
                    existingLoad.push(c);
                }
            }
        }

        // Step A: Unload surplus ships from existing carriers onto the star first.
        for (const c of existingUnload) {
            const cid = c._id.toString();
            const before = beforeCarriers.get(cid)!;
            const afterShips = Number(c.ships ?? 1);
            const deltaToStar = before.ships - afterShips;
            runningStarShips += deltaToStar;
            unloadTransfers.push({
                type: "TransferShips",
                carrierId: cid,
                starId,
                carrierShips: afterShips,
                starShips: runningStarShips,
            });
        }

        // Step B: Build new carriers directly with their final ship count!
        for (const c of newlyBuilt) {
            const tempId = c._id.toString();
            const ships = Math.max(1, Number(c.ships ?? 1));
            runningStarShips = Math.max(0, runningStarShips - ships);
            buildCarrierActions.push({
                type: "BuildCarrier",
                tempCarrierId: tempId,
                starId,
                starName: starBefore.name,
                ships,
            });
        }

        // Step C: Load ships from the star onto existing carriers whose ship count grew.
        for (const c of existingLoad) {
            const cid = c._id.toString();
            const before = beforeCarriers.get(cid)!;
            const afterShips = Number(c.ships ?? 1);
            const deltaFromStar = afterShips - before.ships;
            runningStarShips = Math.max(0, runningStarShips - deltaFromStar);
            loadTransfers.push({
                type: "TransferShips",
                carrierId: cid,
                starId,
                carrierShips: afterShips,
                starShips: runningStarShips,
            });
        }
    }

    // Stage 5 (`HireCarrierSpecialist`) & Stage 6 (`SaveWaypoints`):
    for (const c of shadowGame.galaxy.carriers) {
        if (c.ownedByPlayerId?.toString() !== myId) continue;
        const cid = c._id.toString();
        if (pinnedCarrierIds.has(cid) || c.isGift) continue;

        const before = beforeCarriers.get(cid);
        const afterSpec =
            c.specialistId != null ? Number(c.specialistId) : null;
        if (afterSpec != null && afterSpec !== (before?.specialistId ?? null)) {
            carrierHireActions.push({
                type: "HireCarrierSpecialist",
                carrierId: cid,
                specialistId: afterSpec,
            });
        }

        const wps = normalizeWaypoints(c.waypoints);
        const afterHash = hashWaypoints(wps);
        if (!before) {
            if (wps.length > 0) {
                saveWaypointActions.push({
                    type: "SaveWaypoints",
                    carrierId: cid,
                    waypoints: wps,
                    looped: Boolean(c.waypointsLooped),
                    isDefense: wps[0]?.action === "dropAll",
                });
            }
        } else if (afterHash !== before.waypointsHash) {
            saveWaypointActions.push({
                type: "SaveWaypoints",
                carrierId: cid,
                waypoints: wps,
                looped: Boolean(c.waypointsLooped),
                isDefense: wps[0]?.action === "dropAll",
            });
        }
    }

    const actions: LiveAction[] = [
        ...params.researchActions,
        ...peaceActions,
        ...starHireActions,
        ...infraActions,
        ...unloadTransfers,
        ...buildCarrierActions,
        ...loadTransfers,
        ...carrierHireActions,
        ...saveWaypointActions,
    ];

    return {
        actions,
        postShadowGame: shadowGame,
        alerts,
    };
}

export function buildIdempotencyKey(
    gameId: string,
    tick: number,
    seq: number,
    action: LiveAction,
    batchScope = "tick",
): string {
    const digest = crypto
        .createHash("sha1")
        .update(JSON.stringify(action))
        .digest("hex")
        .slice(0, 10);
    return `${gameId}-t${tick}-${batchScope}-${seq}-${digest}`;
}

/**
 * Sequentially dispatches `LiveAction[]` through `SolarisLiveClient` (concurrency = 1)
 * with:
 * - Tick-deadline guard (`nextTickDate - now > estimatedBatchMs + deadlineSafetyMs`),
 * - Deterministic `Idempotency-Key` headers (skipping keys already dispatched before a crash),
 * - `tempCarrierId -> serverCarrierId` remapping on `BuildCarrier`, and
 * - Stale-state `400 ValidationError` conflict recovery (aborting remaining batch cleanly).
 */
export async function dispatchLiveActions(params: {
    client: SolarisLiveClient;
    gameId: string;
    tick: number;
    actions: LiveAction[];
    batchScope?: string;
    nextTickDate?: string | Date | null;
    maxRequestsPerSecond: number;
    deadlineSafetyMs: number;
    dispatchedKeys: Set<string>;
    onKeyDispatched?: (key: string) => void;
    onAlert?: (alert: LiveAlert) => void;
}): Promise<DispatchBatchResult> {
    const { client, gameId, tick, dispatchedKeys } = params;
    const batchScope = params.batchScope ?? "tick";
    const remappedCarrierIds = new Map<string, string>();
    let actionsToRun = params.actions;
    let deferredByDeadline = false;

    if (!actionsToRun.length) {
        return {
            dispatchedCount: 0,
            skippedDuplicateCount: 0,
            deferredByDeadline: false,
            abortedOnConflict: false,
            remappedCarrierIds,
        };
    }

    // Tick-Deadline Guard (Section 2.4):
    if (params.nextTickDate) {
        const nextMs = new Date(params.nextTickDate).getTime();
        if (Number.isFinite(nextMs)) {
            const remMs = nextMs - Date.now();
            const estMs =
                (actionsToRun.length /
                    Math.max(0.5, params.maxRequestsPerSecond)) *
                1000;
            if (remMs > 0 && remMs < estMs + params.deadlineSafetyMs) {
                // Too close to tick lock: emit ONLY urgent research + peace/defense actions!
                deferredByDeadline = true;
                actionsToRun = actionsToRun.filter(
                    (a) =>
                        a.type === "SetResearchNow" ||
                        a.type === "SetResearchQueue" ||
                        (a.type === "SaveWaypoints" && a.isDefense),
                );
                params.onAlert?.({
                    timestamp: new Date().toISOString(),
                    tick,
                    severity: "warn",
                    code: "TICK_DEADLINE_GUARD",
                    message: `Tick deadline approaching (${Math.round(remMs / 1000)}s left); deferred ${params.actions.length - actionsToRun.length} non-urgent action(s) to next tick.`,
                });
            }
        }
    }

    let dispatchedCount = 0;
    let skippedDuplicateCount = 0;

    const resolveCarrierId = (id: string): string =>
        remappedCarrierIds.get(id) ?? id;

    for (let seq = 0; seq < actionsToRun.length; seq++) {
        const rawAction = actionsToRun[seq];
        const action: LiveAction =
            rawAction.type === "TransferShips"
                ? {
                      ...rawAction,
                      carrierId: resolveCarrierId(rawAction.carrierId),
                  }
                : rawAction.type === "HireCarrierSpecialist"
                  ? {
                        ...rawAction,
                        carrierId: resolveCarrierId(rawAction.carrierId),
                    }
                  : rawAction.type === "SaveWaypoints"
                    ? {
                          ...rawAction,
                          carrierId: resolveCarrierId(rawAction.carrierId),
                      }
                    : rawAction;

        const idemKey = buildIdempotencyKey(
            gameId,
            tick,
            seq,
            action,
            batchScope,
        );
        if (dispatchedKeys.has(idemKey) && action.type !== "BuildCarrier") {
            skippedDuplicateCount++;
            continue;
        }

        try {
            switch (action.type) {
                case "SetResearchNow":
                    await client.updateResearchNow(
                        gameId,
                        action.preference,
                        idemKey,
                    );
                    break;
                case "SetResearchQueue":
                    await client.updateResearchQueue(
                        gameId,
                        action.queue,
                        idemKey,
                    );
                    break;
                case "HireStarSpecialist":
                    await client.hireStarSpecialist(
                        gameId,
                        action.starId,
                        action.specialistId,
                        idemKey,
                    );
                    break;
                case "UpgradeStar":
                    await client.upgradeStar(
                        gameId,
                        action.starId,
                        action.infrastructure,
                        idemKey,
                    );
                    break;
                case "BuildWarpGate":
                    await client.buildWarpGate(gameId, action.starId, idemKey);
                    break;
                case "BuildCarrier": {
                    const report = await client.buildCarrier(
                        gameId,
                        action.starId,
                        action.ships,
                        idemKey,
                    );
                    const realId = report?.carrier?._id?.toString();
                    if (!realId) {
                        throw new Error(
                            "BuildCarrier response did not include carrier._id",
                        );
                    }
                    remappedCarrierIds.set(action.tempCarrierId, realId);
                    break;
                }
                case "TransferShips":
                    await client.transferShips(
                        gameId,
                        action.carrierId,
                        action.starId,
                        action.carrierShips,
                        action.starShips,
                        idemKey,
                    );
                    break;
                case "HireCarrierSpecialist":
                    await client.hireCarrierSpecialist(
                        gameId,
                        action.carrierId,
                        action.specialistId,
                        idemKey,
                    );
                    break;
                case "SaveWaypoints":
                    await client.saveWaypoints(
                        gameId,
                        action.carrierId,
                        action.waypoints,
                        action.looped,
                        idemKey,
                    );
                    break;
            }

            dispatchedKeys.add(idemKey);
            params.onKeyDispatched?.(idemKey);
            dispatchedCount++;
        } catch (err: any) {
            if (err instanceof SolarisHttpError && err.status === 409) {
                // Duplicate Idempotency-Key accepted earlier before a network hiccup.
                dispatchedKeys.add(idemKey);
                skippedDuplicateCount++;
                continue;
            }
            if (err instanceof SolarisHttpError && err.status === 400) {
                const msg = `Stale-state 400 conflict on ${action.type} (${err.bodyText}); aborting remaining ${actionsToRun.length - seq - 1} dependent action(s) and re-syncing galaxy.`;
                params.onAlert?.({
                    timestamp: new Date().toISOString(),
                    tick,
                    severity: "warn",
                    code: "STALE_STATE_CONFLICT",
                    message: msg,
                    details: {
                        action,
                        status: err.status,
                        error: err.bodyText,
                    },
                });
                return {
                    dispatchedCount,
                    skippedDuplicateCount,
                    deferredByDeadline,
                    abortedOnConflict: true,
                    conflictError: err.bodyText,
                    remappedCarrierIds,
                };
            }
            throw err;
        }
    }

    return {
        dispatchedCount,
        skippedDuplicateCount,
        deferredByDeadline,
        abortedOnConflict: false,
        remappedCarrierIds,
    };
}
