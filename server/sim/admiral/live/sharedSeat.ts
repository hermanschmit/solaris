import { SimEngine } from "../../engine";
import { Game } from "../../../services/types/Game";
import { Player } from "../../../services/types/Player";
import {
    CarrierCommitSnapshot,
    HumanState,
    PinnedCarrierState,
    WaypointSpec,
} from "./types";

const MANUAL_NAME_RE = /^(\[H\]|MANUAL\b)/i;

export function normalizeWaypoints(
    waypoints: any[] | undefined,
): WaypointSpec[] {
    if (!waypoints?.length) return [];
    return waypoints.map((w) => ({
        source: w.source.toString(),
        destination: w.destination.toString(),
        action: (w.action ?? "nothing") as WaypointSpec["action"],
        actionShips: Number(w.actionShips ?? 0),
        delayTicks: Number(w.delayTicks ?? 0),
    }));
}

export function hashWaypoints(waypoints: WaypointSpec[]): string {
    if (!waypoints.length) return "";
    return waypoints
        .map(
            (w) =>
                `${w.source}->${w.destination}:${w.action}:${w.actionShips}:${w.delayTicks}`,
        )
        .join("|");
}

export function snapshotCarrier(
    carrier: any,
    tick: number,
): CarrierCommitSnapshot {
    const wps = normalizeWaypoints(carrier.waypoints);
    return {
        carrierId: carrier._id.toString(),
        ships: Number(carrier.ships ?? 0),
        orbiting: carrier.orbiting ? carrier.orbiting.toString() : null,
        waypoints: wps,
        waypointsHash: hashWaypoints(wps),
        isGift: Boolean(carrier.isGift),
        looped: Boolean(carrier.waypointsLooped),
        specialistId:
            carrier.specialistId != null ? Number(carrier.specialistId) : null,
        tick,
    };
}

/**
 * Checks whether `current` is the natural server tick progression of an
 * AI-committed carrier `prev` (e.g. advancing along its plotted route,
 * decrementing `delayTicks`, or finishing its route into idle orbit), as
 * opposed to a manual human edit in the Solaris GUI.
 */
export function isNaturalProgressionOfAiCommit(
    prev: CarrierCommitSnapshot,
    current: any,
    ticksElapsed: number,
): boolean {
    if (Boolean(current.isGift) !== prev.isGift) return false;
    if (Boolean(current.waypointsLooped) !== prev.looped) return false;

    const curWps = normalizeWaypoints(current.waypoints);
    const curHash = hashWaypoints(curWps);

    // Same tick (e.g. reactive mid-tick wake-up): must match exact commit!
    if (ticksElapsed <= 0) {
        return (
            curHash === prev.waypointsHash &&
            Number(current.ships ?? 0) === prev.ships &&
            (current.orbiting ? current.orbiting.toString() : null) ===
                prev.orbiting
        );
    }

    // Previously idle orbiting AI carrier: it never moves or changes ship count
    // on its own unless someone transferred ships or plotted waypoints in GUI!
    if (prev.waypoints.length === 0) {
        return (
            curWps.length === 0 &&
            Number(current.ships ?? 0) === prev.ships &&
            (current.orbiting ? current.orbiting.toString() : null) ===
                prev.orbiting
        );
    }

    // Previously active AI carrier:
    // Case A: Finished all waypoints and now sits idle at final destination.
    if (curWps.length === 0) {
        const lastDest = prev.waypoints[prev.waypoints.length - 1].destination;
        const curOrbit = current.orbiting ? current.orbiting.toString() : null;
        return curOrbit === lastDest;
    }

    // Case B: Still executing a suffix of `prev.waypoints`!
    if (curWps.length > prev.waypoints.length) return false;
    const offset = prev.waypoints.length - curWps.length;
    for (let i = 0; i < curWps.length; i++) {
        const pWp = prev.waypoints[offset + i];
        const cWp = curWps[i];
        if (
            cWp.source !== pWp.source ||
            cWp.destination !== pWp.destination ||
            cWp.action !== pWp.action ||
            cWp.actionShips !== pWp.actionShips
        ) {
            return false;
        }
        // First remaining hop may have decremented `delayTicks`, never increased!
        if (i === 0) {
            if (cWp.delayTicks > pWp.delayTicks) return false;
        } else if (cWp.delayTicks !== pWp.delayTicks) {
            return false;
        }
    }

    // If still orbiting at the starting star before launch (`offset === 0` and
    // `current.orbiting === prev.orbiting`), ship count shouldn't change unless
    // the human manually transferred ships!
    const curOrbit = current.orbiting ? current.orbiting.toString() : null;
    if (
        offset === 0 &&
        prev.orbiting != null &&
        curOrbit === prev.orbiting &&
        Number(current.ships ?? 0) !== prev.ships
    ) {
        return false;
    }

    return true;
}

/**
 * Updates `human.humanPinnedCarriers` and returns the active `Set<string>` of
 * carrier IDs that Admiral MUST NOT pool ships off, reroute, or hire specialists on.
 *
 * On initial daemon attach (`hasPriorBaseline === false`), existing normal idle/en-route
 * carriers without `isGift` / `looped` / `[H]` prefix are seeded into `aiCommittedCarriers`
 * rather than falsely treated as GUI-created; on every subsequent tick/refresh, any
 * newly created carrier or manually modified carrier is automatically pinned with an
 * idle grace window (`graceTicks`, default 2 ticks after becoming idle).
 */
export function updateCarrierPins(params: {
    ownCarriers: any[];
    human: HumanState;
    tick: number;
    lastProcessedTick: number;
    graceTicks?: number;
}): Set<string> {
    const { ownCarriers, human, tick, lastProcessedTick } = params;
    const graceTicks = Math.max(1, params.graceTicks ?? 2);
    const hasPriorBaseline = lastProcessedTick >= 0;
    const tickAdvanced = lastProcessedTick >= 0 && tick > lastProcessedTick;
    const ticksElapsed =
        lastProcessedTick >= 0 ? Math.max(0, tick - lastProcessedTick) : 0;

    const liveIds = new Set<string>();

    for (const c of ownCarriers) {
        const cid = c._id.toString();
        liveIds.add(cid);

        const isGift = Boolean(c.isGift);
        const isLooped = Boolean(c.waypointsLooped);
        const isManualPrefix = MANUAL_NAME_RE.test(String(c.name ?? ""));
        const hasWaypoints = Boolean(c.waypoints && c.waypoints.length > 0);

        let pinReason: PinnedCarrierState["reason"] | null = null;
        if (isGift) pinReason = "gift";
        else if (isLooped) pinReason = "looped";
        else if (isManualPrefix) pinReason = "manual_prefix";

        const existingPin = human.humanPinnedCarriers[cid];
        const aiSnap = human.aiCommittedCarriers[cid];

        if (!pinReason && !existingPin) {
            if (!hasPriorBaseline) {
                // First attach: seed baseline snapshot so Admiral manages normal carriers.
                human.aiCommittedCarriers[cid] = snapshotCarrier(c, tick);
            } else if (!aiSnap) {
                // Carrier appeared after Admiral's last run -> built by human in GUI!
                pinReason = "gui_created";
            } else if (
                !isNaturalProgressionOfAiCommit(aiSnap, c, ticksElapsed)
            ) {
                // Carrier differs from Admiral's committed state -> edited by human in GUI!
                pinReason = "gui_modified";
                delete human.aiCommittedCarriers[cid];
            }
        }

        if (pinReason) {
            human.humanPinnedCarriers[cid] = {
                carrierId: cid,
                reason: pinReason,
                pinnedAtTick: existingPin?.pinnedAtTick ?? tick,
                idleGraceRemaining: graceTicks,
            };
            delete human.aiCommittedCarriers[cid];
            continue;
        }

        if (existingPin) {
            if (hasWaypoints || isGift || isLooped || isManualPrefix) {
                existingPin.idleGraceRemaining = graceTicks;
            } else if (tickAdvanced) {
                existingPin.idleGraceRemaining -= 1;
                if (existingPin.idleGraceRemaining < 0) {
                    delete human.humanPinnedCarriers[cid];
                    human.aiCommittedCarriers[cid] = snapshotCarrier(c, tick);
                }
            }
        }
    }

    // Prune carriers that no longer exist (destroyed, arrived gift, merged, etc.).
    for (const cid of Object.keys(human.humanPinnedCarriers)) {
        if (!liveIds.has(cid)) {
            delete human.humanPinnedCarriers[cid];
        }
    }
    for (const cid of Object.keys(human.aiCommittedCarriers)) {
        if (!liveIds.has(cid)) {
            delete human.aiCommittedCarriers[cid];
        }
    }

    return new Set(Object.keys(human.humanPinnedCarriers));
}

/**
 * Records Admiral's post-execution carrier state into `human.aiCommittedCarriers`,
 * remapping any shadow `tempId`s to real server carrier IDs returned by `BuildCarrier`.
 */
export function recordAiCommittedCarriers(params: {
    postShadowCarriers: any[];
    myPlayerId: string;
    pinnedCarrierIds: ReadonlySet<string>;
    remappedCarrierIds: ReadonlyMap<string, string>;
    unmappedTempCarrierIds?: ReadonlySet<string>;
    human: HumanState;
    tick: number;
}) {
    const {
        postShadowCarriers,
        myPlayerId,
        pinnedCarrierIds,
        remappedCarrierIds,
        unmappedTempCarrierIds,
        human,
        tick,
    } = params;
    const nextCommits: Record<string, CarrierCommitSnapshot> = {};

    for (const c of postShadowCarriers) {
        if (c.ownedByPlayerId?.toString() !== myPlayerId) continue;
        const rawId = c._id.toString();
        const realId = remappedCarrierIds.get(rawId) ?? rawId;
        if (pinnedCarrierIds.has(realId) || pinnedCarrierIds.has(rawId)) {
            continue;
        }
        if (
            unmappedTempCarrierIds?.has(rawId) &&
            !remappedCarrierIds.has(rawId)
        ) {
            continue;
        }
        const snap = snapshotCarrier(c, tick);
        snap.carrierId = realId;
        nextCommits[realId] = snap;
    }

    human.aiCommittedCarriers = nextCommits;
}

/**
 * Computes credits needed by any active GUI `player.scheduledActions` scheduled
 * during the current production cycle so Admiral reserves those credits.
 */
export function estimateScheduledActionsReserve(
    engine: SimEngine,
    game: Game,
    player: Player,
): number {
    const actions = (player as any).scheduledActions as any[] | undefined;
    if (!actions?.length) return 0;

    const tick = game.state.tick;
    const prodTicks = game.settings.galaxy.productionTicks || 20;
    const cycleEnd = (Math.floor(tick / prodTicks) + 1) * prodTicks;
    let total = 0;

    for (const act of actions) {
        if (act.active === false) continue;
        const actTick = Number(act.tick ?? 0);
        if (!act.repeat && (actTick < tick || actTick > cycleEnd)) {
            continue;
        }
        const buyType = String(act.buyType ?? "totalCredits");
        const amount = Number(act.amount ?? 0);
        if (amount <= 0) continue;

        if (buyType === "totalCredits" || buyType === "credits") {
            total += amount;
        } else if (buyType === "percentageOfCredits") {
            total += Math.ceil(
                ((player.credits ?? 0) * Math.min(100, amount)) / 100,
            );
        } else {
            try {
                const clone = engine.clone(game);
                const cp = clone.galaxy.players.find(
                    (p) => p._id.toString() === player._id.toString(),
                )!;
                const report =
                    engine.container.starUpgradeService.generateUpgradeBulkReport(
                        clone,
                        cp,
                        buyType,
                        act.infrastructureType,
                        amount,
                    );
                total += Number(report?.cost ?? 0);
            } catch {
                total += amount * 25;
            }
        }
    }

    return Math.max(0, Math.ceil(total));
}
