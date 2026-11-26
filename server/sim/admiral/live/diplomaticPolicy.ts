import { DiplomaticStatus } from "@solaris/common";
import { TacticalDiplomacyHooks } from "../tacticalExecutor";
import { parseAttackTargetSpec } from "./notesParser";
import {
    DiplomaticTier,
    LiveAction,
    LiveAlert,
    NeutralBorderPosture,
    WaypointSpec,
} from "./types";

export interface DiplomaticPlayerEntry {
    playerId: string;
    alias: string;
    statusTo: "allies" | "neutral" | "enemies";
    statusFrom: "allies" | "neutral" | "enemies";
    actualStatus: "allies" | "neutral" | "enemies";
    tier: DiplomaticTier;
    covertAttack: boolean;
    attackStartTick: number | null;
}

export interface DiplomaticPolicy {
    myPlayerId: string;
    entriesByPlayerId: Map<string, DiplomaticPlayerEntry>;
    tiersBySeatIndex: DiplomaticTier[];
    neutralBorderPosture: NeutralBorderPosture;
    getTier: (playerId: string) => DiplomaticTier;
    isAllied: (playerId: string) => boolean;
    isProtected: (playerId: string) => boolean;
    isHostile: (playerId: string) => boolean;
    getAttackStartTick: (playerId: string) => number | null;
    toExecutorHooks: () => TacticalDiplomacyHooks;
}

/**
 * Resolves the 3-tier diplomatic posture (`ALLIED` / `PROTECTED` / `HOSTILE`)
 * from Solaris GUI diplomacy (`GET /api/game/:gameId/diplomacy`) plus optional
 * private `attack:` overrides (including scheduled `PlayerAlias@<tick>`):
 *
 * - `ALLIED`: `actualStatus === "allies"` — ALWAYS wins; listing a formal ally
 *   in `attack:` is ignored with an explicit warning alert.
 * - `PROTECTED`: `actualStatus === "neutral"` (when not in `attack:`) OR
 *   pending alliance offer (`statusTo === "allies"` and `actualStatus === "neutral"`).
 * - `HOSTILE`: `actualStatus === "enemies"` OR (`actualStatus === "neutral"`,
 *   `statusTo !== "allies"`, and matched by ID or alias in `attack:`).
 */
export function buildDiplomaticPolicy(params: {
    myPlayerId: string;
    players: { _id: any; alias: string; diplomacy?: any[] }[];
    diplomaticStatuses?: DiplomaticStatus<string>[] | null;
    attackList?: string[];
    neutralBorderPosture?: NeutralBorderPosture;
    tick: number;
    onAlert?: (alert: LiveAlert) => void;
}): DiplomaticPolicy {
    const myId = params.myPlayerId.toString();
    const neutralBorderPosture = params.neutralBorderPosture ?? "defend";
    const statusMap = new Map<string, DiplomaticStatus<string>>();
    for (const st of params.diplomaticStatuses ?? []) {
        statusMap.set(st.playerIdTo.toString(), st);
    }

    const me = params.players.find((p) => p._id.toString() === myId);
    const attackScheduleByKey = new Map<string, number | null>();
    for (const raw of params.attackList ?? []) {
        const spec = parseAttackTargetSpec(raw);
        if (!spec) continue;
        const key = spec.target.toLowerCase();
        if (!attackScheduleByKey.has(key)) {
            attackScheduleByKey.set(key, spec.startTick);
        } else {
            const prev = attackScheduleByKey.get(key)!;
            if (prev === null || spec.startTick === null) {
                attackScheduleByKey.set(key, null);
            } else {
                attackScheduleByKey.set(key, Math.min(prev, spec.startTick));
            }
        }
    }

    const entries = new Map<string, DiplomaticPlayerEntry>();
    const tiersBySeatIndex: DiplomaticTier[] = [];

    for (const p of params.players) {
        const pid = p._id.toString();
        const alias = p.alias ?? pid;

        if (pid === myId) {
            entries.set(pid, {
                playerId: pid,
                alias,
                statusTo: "allies",
                statusFrom: "allies",
                actualStatus: "allies",
                tier: "ALLIED",
                covertAttack: false,
                attackStartTick: null,
            });
            tiersBySeatIndex.push("ALLIED");
            continue;
        }

        const apiSt = statusMap.get(pid);
        const statusTo: "allies" | "neutral" | "enemies" =
            (apiSt?.statusTo as any) ??
            me?.diplomacy?.find((d: any) => d.playerId?.toString() === pid)
                ?.status ??
            "neutral";
        const statusFrom: "allies" | "neutral" | "enemies" =
            (apiSt?.statusFrom as any) ??
            p.diplomacy?.find((d: any) => d.playerId?.toString() === myId)
                ?.status ??
            "neutral";
        const actualStatus: "allies" | "neutral" | "enemies" =
            (apiSt?.actualStatus as any) ??
            (statusTo === "enemies" || statusFrom === "enemies"
                ? "enemies"
                : statusTo === "allies" && statusFrom === "allies"
                  ? "allies"
                  : "neutral");

        const pidKey = pid.toLowerCase();
        const aliasKey = alias.trim().toLowerCase();
        const matchedByPid = attackScheduleByKey.has(pidKey);
        const matchedByAlias = attackScheduleByKey.has(aliasKey);
        const matchedInAttack = matchedByPid || matchedByAlias;
        let scheduledStartTick: number | null = null;
        if (matchedByPid && matchedByAlias) {
            const t1 = attackScheduleByKey.get(pidKey)!;
            const t2 = attackScheduleByKey.get(aliasKey)!;
            scheduledStartTick =
                t1 === null || t2 === null ? null : Math.min(t1, t2);
        } else if (matchedByPid) {
            scheduledStartTick = attackScheduleByKey.get(pidKey)!;
        } else if (matchedByAlias) {
            scheduledStartTick = attackScheduleByKey.get(aliasKey)!;
        }

        let tier: DiplomaticTier;
        let covertAttack = false;
        let attackStartTick: number | null = null;

        if (actualStatus === "allies") {
            tier = "ALLIED";
            if (matchedInAttack) {
                params.onAlert?.({
                    timestamp: new Date().toISOString(),
                    tick: params.tick,
                    severity: "warn",
                    code: "ALLY_IN_ATTACK_IGNORED",
                    message: `Ignored private 'attack' override for formal ally '${alias}'. Break alliance in the GUI first if you intend war.`,
                    details: { playerId: pid, alias },
                });
            }
        } else if (actualStatus === "enemies") {
            tier = "HOSTILE";
        } else if (statusTo === "allies") {
            // Pending alliance offer from us: strictly PROTECTED.
            tier = "PROTECTED";
        } else if (matchedInAttack) {
            tier = "HOSTILE";
            covertAttack = true;
            attackStartTick = scheduledStartTick;
        } else {
            tier = "PROTECTED";
        }

        entries.set(pid, {
            playerId: pid,
            alias,
            statusTo,
            statusFrom,
            actualStatus,
            tier,
            covertAttack,
            attackStartTick,
        });
        tiersBySeatIndex.push(tier);
    }

    const getTier = (playerId: string): DiplomaticTier => {
        if (playerId === myId) return "ALLIED";
        return entries.get(playerId)?.tier ?? "PROTECTED";
    };

    const isAllied = (playerId: string) => getTier(playerId) === "ALLIED";
    const isProtected = (playerId: string) => getTier(playerId) === "PROTECTED";
    const isHostile = (playerId: string) => getTier(playerId) === "HOSTILE";
    const getAttackStartTick = (playerId: string): number | null =>
        entries.get(playerId)?.attackStartTick ?? null;

    return {
        myPlayerId: myId,
        entriesByPlayerId: entries,
        tiersBySeatIndex,
        neutralBorderPosture,
        getTier,
        isAllied,
        isProtected,
        isHostile,
        getAttackStartTick,
        toExecutorHooks: () => ({
            isHostile,
            isAllied,
            isProtected,
            getAttackStartTick,
            neutralBorderPosture,
            alliedBorderPosture: "open",
            allowAlliedStarTransit: false,
            allowContestedNeutralClaims: false,
        }),
    };
}

/**
 * Invariant 2 — Mid-Flight & Delayed-Launch Peace / Alliance Handler:
 * - Orbiting carriers with waypoints targeting `ALLIED` or `PROTECTED` stars
 *   (such as `delayTicks > 0` staged ToT invasions) immediately clear waypoints.
 * - In-flight carriers whose first hop heads to an `ALLIED` or `PROTECTED` star
 *   cannot alter `waypoints[0]` in Solaris (`saveWaypoints.ts:85-105`), so we:
 *     1. Truncate any `waypoints[1..]` / `looped` continuation, and
 *     2. Emit an `UNRECALLABLE_FLEET` urgent alert with carrier, ships, target, ETA.
 * - In-flight carriers whose `waypoints[0]` is safe but a later queued hop `k >= 1`
 *   targets an `ALLIED` or `PROTECTED` star truncate `waypoints` to `0..k-1`.
 */
export function buildPeaceCancellationActions(params: {
    game: any;
    policy: DiplomaticPolicy;
    pinnedCarrierIds?: ReadonlySet<string>;
    tick: number;
    calculateEta?: (carrier: any, wp: any) => number;
    onAlert?: (alert: LiveAlert) => void;
    mutateShadow?: boolean;
}): LiveAction[] {
    const { game, policy, tick } = params;
    const myId = policy.myPlayerId;
    const starById = new Map<string, any>();
    for (const s of game.galaxy.stars ?? []) {
        starById.set(s._id.toString(), s);
    }

    const isForbiddenTargetStar = (
        starId: string,
    ): {
        forbidden: boolean;
        star: any;
        ownerId: string | null;
        tier: DiplomaticTier | null;
    } => {
        const star = starById.get(starId);
        const ownerId = star?.ownedByPlayerId?.toString() ?? null;
        if (!ownerId || ownerId === myId) {
            return { forbidden: false, star, ownerId, tier: null };
        }
        const tier = policy.getTier(ownerId);
        return {
            forbidden: tier === "ALLIED" || tier === "PROTECTED",
            star,
            ownerId,
            tier,
        };
    };

    const actions: LiveAction[] = [];

    for (const c of game.galaxy.carriers ?? []) {
        if (c.ownedByPlayerId?.toString() !== myId) continue;
        if (c.isGift || c.isScuttled) continue;
        const cid = c._id.toString();
        if (params.pinnedCarrierIds?.has(cid)) continue;
        if (!c.waypoints?.length) continue;

        if (c.orbiting != null) {
            const badHop = c.waypoints.find((wp: any) => {
                const chk = isForbiddenTargetStar(wp.destination.toString());
                return chk.forbidden;
            });
            if (badHop) {
                const chk = isForbiddenTargetStar(
                    badHop.destination.toString(),
                );
                const targetAlias =
                    (chk.ownerId &&
                        policy.entriesByPlayerId.get(chk.ownerId)?.alias) ??
                    chk.ownerId ??
                    "non-hostile";
                actions.push({
                    type: "SaveWaypoints",
                    carrierId: cid,
                    waypoints: [],
                    looped: false,
                    reason: "peace_cancel_orbiting",
                    isDefense: true,
                });
                if (params.mutateShadow) {
                    c.waypoints = [];
                    c.waypointsLooped = false;
                }
                params.onAlert?.({
                    timestamp: new Date().toISOString(),
                    tick,
                    severity: "info",
                    code: "PEACE_WAYPOINTS_CANCELLED",
                    message: `Cancelled staged waypoints on orbiting carrier '${c.name}' (${c.ships ?? 0} ships) toward ${chk.tier} player '${targetAlias}' (${chk.star?.name ?? badHop.destination}).`,
                    details: {
                        carrierId: cid,
                        carrierName: c.name,
                        ships: c.ships ?? 0,
                        destinationStarId: badHop.destination.toString(),
                        targetPlayerId: chk.ownerId,
                        tier: chk.tier,
                    },
                });
            }
            continue;
        }

        // Carrier is mid-flight (`c.orbiting == null`)
        const wp0 = c.waypoints[0];
        const firstChk = isForbiddenTargetStar(wp0.destination.toString());
        if (firstChk.forbidden) {
            const targetAlias =
                (firstChk.ownerId &&
                    policy.entriesByPlayerId.get(firstChk.ownerId)?.alias) ??
                firstChk.ownerId ??
                "non-hostile";
            const eta = wp0.ticksEta ?? params.calculateEta?.(c, wp0) ?? 1;

            if (c.waypoints.length > 1 || c.waypointsLooped) {
                const keptWp0: WaypointSpec = {
                    source: wp0.source.toString(),
                    destination: wp0.destination.toString(),
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 0,
                };
                actions.push({
                    type: "SaveWaypoints",
                    carrierId: cid,
                    waypoints: [keptWp0],
                    looped: false,
                    reason: "peace_truncate_inflight",
                    isDefense: true,
                });
                if (params.mutateShadow) {
                    c.waypoints = [c.waypoints[0]];
                    c.waypoints[0].delayTicks = 0;
                    c.waypointsLooped = false;
                }
            }

            params.onAlert?.({
                timestamp: new Date().toISOString(),
                tick,
                severity: "urgent",
                code: "UNRECALLABLE_FLEET",
                message: `URGENT: In-flight carrier '${c.name}' (${c.ships ?? 0} ships) cannot be recalled mid-warp and will arrive at ${firstChk.star?.name ?? wp0.destination} (${firstChk.tier}: ${targetAlias}) in ${eta} tick(s)! Message ${targetAlias}, finalize formal alliance, or gift the carrier in the GUI before arrival.`,
                details: {
                    carrierId: cid,
                    carrierName: c.name,
                    ships: c.ships ?? 0,
                    destinationStarId: wp0.destination.toString(),
                    destinationStarName: firstChk.star?.name,
                    targetPlayerId: firstChk.ownerId,
                    targetAlias,
                    tier: firstChk.tier,
                    eta,
                },
            });
            continue;
        }

        // First hop is safe; check if any subsequent queued hop hits ALLIED / PROTECTED.
        let cutAt = -1;
        for (let i = 1; i < c.waypoints.length; i++) {
            if (
                isForbiddenTargetStar(c.waypoints[i].destination.toString())
                    .forbidden
            ) {
                cutAt = i;
                break;
            }
        }
        if (cutAt >= 1) {
            const truncated: WaypointSpec[] = c.waypoints
                .slice(0, cutAt)
                .map((w: any, idx: number) => ({
                    source: w.source.toString(),
                    destination: w.destination.toString(),
                    action: (w.action ?? "nothing") as WaypointSpec["action"],
                    actionShips: w.actionShips ?? 0,
                    delayTicks: idx === 0 ? 0 : (w.delayTicks ?? 0),
                }));
            actions.push({
                type: "SaveWaypoints",
                carrierId: cid,
                waypoints: truncated,
                looped: false,
                reason: "peace_truncate_queued_hops",
                isDefense: true,
            });
            if (params.mutateShadow) {
                c.waypoints = c.waypoints.slice(0, cutAt);
                c.waypoints[0].delayTicks = 0;
                c.waypointsLooped = false;
            }
        }
    }

    return actions;
}
