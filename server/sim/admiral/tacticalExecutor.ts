// Purpose-built tactical executor for Admiral seats.
//
// Replaces AIService's single-star greedy heuristic with an executor designed
// around continuous spatial intent from a neural policy + exact algorithmic
// micro:
//   1. Triage & evacuation: evaluates incoming enemy fleets; reinforces stars
//      that can be held and evacuates garrisons from doomed stars before impact
//      (never buying upgrades on doomed stars).
//   2. Targeted pointer upgrades + Warp Gates: spends cycle budgets by
//      (priority / marginal_cost) across stars for Science, Industry, Economy
//      and arterial Warp Gates.
//   3. Multi-hop & Explorer claims: expands to unclaimed stars across the
//      hyperspace graph and hires Explorer specialists for out-of-range pockets.
//   4. Synchronized Time-on-Target (ToT) multi-star invasions: pools surplus
//      ships from multiple nearby stars and sets `delayTicks` on closer
//      carriers so all attacking fleets arrive on the exact same tick.
//   5. Proactive border garrisoning & Warp-Gate logistics: routes interior
//      ships above `garrisonTarget` along shortest tick-cost paths toward
//      under-garrisoned chokepoints and staging stars.
//
// Runs strictly inside `applyFog` when fog of war is active, so it never reads
// hidden enemy carriers, hidden infrastructure or true hidden garrisons.

import mongoose from "mongoose";
import { CarrierWaypointBase, ValidationError } from "@solaris/common";
import { VISIBLE } from "../beliefData";
import { Game } from "../../services/types/Game";
import { Player } from "../../services/types/Player";
import { Star, InfrastructureType } from "../../services/types/Star";
import { Carrier } from "../../services/types/Carrier";
import { DBObjectId } from "../../services/types/DBObjectId";
import { DependencyContainer } from "../../services/types/DependencyContainer";

const SPEC_EXPLORER = 6; // +3 hyperspace, x1.5 speed, -3 weapons (cost 3)
const SPEC_INFILTRATOR = 9; // -1 enemy weapons, no range/speed penalty (cost 3)
const SPEC_SIEGE_BREAKER = 20; // +4 attacker weapons vs star, 15 ticks, no range/speed penalty (cost 3)

export interface AdmiralDecision {
    // Per-star spatial intent (length S):
    // Desired minimum ship garrison on own stars (proactive border deterrence).
    garrisonTarget?: number[];
    // Multiplier / score on non-own stars for claiming & invading (>0).
    attackDesire?: number[];
    // Pointer weights (>0) over own stars for infrastructure & warp gates.
    industryPriority?: number[];
    sciencePriority?: number[];
    economyPriority?: number[];
    warpGatePriority?: number[];

    // Macro cycle budget shares: [science, industry, economy, warpGate, carrierReserve]
    budgetSplit?: number[];
    // Safety multiplier on calculated ships needed for invasions (default 1.35).
    attackMargin?: number;
    // Maximum synchronized arrival horizon in ticks for invasions/claims (default 40).
    maxAttackTicks?: number;
    // Fraction of required defense below which an attacked star is triaged as doomed
    // and evacuated rather than fed piecemeal reinforcements (default 0.85).
    triageThreshold?: number;

    // Discrete cycle choices:
    research?: string;
    researchNext?: string;
    hire?: { star: number; specialist: number }[];
    autoCarrierSpecialist?: boolean;
    court?: boolean;
    betray?: boolean;

    // Under fog: conservation-constrained believed ships & industry per star.
    beliefs?: {
        ships?: (number | null)[];
        industry?: (number | null)[];
    };
}

interface IncomingWave {
    starIndex: number;
    star: Star;
    ticksUntil: number;
    carriers: Carrier[];
}

export interface TacticalAlertEvent {
    kind:
        | "surprise_attack"
        | "emergency_reserve_dip"
        | "emergency_specialist_dip";
    starIndex: number;
    starId: string;
    starName: string;
    attackerPlayerId?: string;
    attackerAlias?: string;
    specialistId?: number;
    specialistName?: string;
    ships: number;
    ticksUntil?: number;
    carrierIds?: string[];
    message: string;
}

export interface TacticalDiplomacyHooks {
    isHostile: (playerId: string) => boolean;
    isAllied: (playerId: string) => boolean;
    isProtected: (playerId: string) => boolean;
    neutralBorderPosture?: "defend" | "open";
    alliedBorderPosture?: "open" | "watch";
    allowAlliedStarTransit?: boolean;
    allowContestedNeutralClaims?: boolean;
}

export interface TacticalExecutionOptions {
    diplomacy?: TacticalDiplomacyHooks;
    pinnedCarrierIds?: ReadonlySet<string>;
    respectIgnoreBulkUpgrade?: boolean;
    cashReserveCredits?: number;
    scheduledActionCredits?: number;
    allowEmergencyDefenseReserveDip?: boolean;
    specialistReserveTokens?: number;
    allowEmergencySpecialistDip?: boolean;
    onAlert?: (alert: TacticalAlertEvent) => void;
}

export class TacticalExecutor {
    container: DependencyContainer;

    // Cached static pairwise Euclidean distances per game (non-orbital).
    private distCache = new WeakMap<Game, Float64Array>();
    // Policy-controlled credits held across the cycle for `isLastTick` Economy
    // compounding (replacing AIService's rigid `totalEconomy * 5` rule).
    private cycleEcoHold = new WeakMap<Player, number>();
    private activeOptions?: TacticalExecutionOptions;
    private activeBlockedEdges?: Set<string>;
    private activeContestedStars?: Set<number>;

    constructor(container: DependencyContainer) {
        this.container = container;
    }

    private _pairDistances(game: Game): Float64Array {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const orbital = this.container.gameTypeService.isOrbitalMode(game);
        if (!orbital) {
            const cached = this.distCache.get(game);
            if (cached && cached.length === S * S) return cached;
        }
        const dist = new Float64Array(S * S);
        const ds = this.container.distanceService;
        const sds = this.container.starDataService;
        for (let i = 0; i < S; i++) {
            for (let j = i + 1; j < S; j++) {
                const d = sds.isStarPairWormHole(stars[i], stars[j])
                    ? 0
                    : ds.getDistanceBetweenLocations(
                          stars[i].location,
                          stars[j].location,
                      );
                dist[i * S + j] = d;
                dist[j * S + i] = d;
            }
        }
        if (!orbital) this.distCache.set(game, dist);
        return dist;
    }

    private _isEnemy(
        game: Game,
        player: Player,
        otherId: DBObjectId | null | undefined,
    ): boolean {
        if (!otherId) return false;
        const otherStr = otherId.toString();
        if (otherStr === player._id.toString()) return false;
        if (this.activeOptions?.diplomacy) {
            return this.activeOptions.diplomacy.isHostile(otherStr);
        }
        if (this.container.diplomacyService.isFormalAlliancesEnabled(game)) {
            return (
                this.container.diplomacyService.getDiplomaticStatusToPlayer(
                    game,
                    player._id,
                    otherId,
                ).actualStatus !== "allies"
            );
        }
        return true;
    }

    private _isAllied(
        game: Game,
        player: Player,
        otherId: DBObjectId | null | undefined,
    ): boolean {
        if (!otherId) return false;
        const otherStr = otherId.toString();
        if (otherStr === player._id.toString()) return true;
        if (this.activeOptions?.diplomacy) {
            return this.activeOptions.diplomacy.isAllied(otherStr);
        }
        if (this.container.diplomacyService.isFormalAlliancesEnabled(game)) {
            return (
                this.container.diplomacyService.getDiplomaticStatusToPlayer(
                    game,
                    player._id,
                    otherId,
                ).actualStatus === "allies"
            );
        }
        return false;
    }

    private _isCarrierPinned(carrier: Carrier): boolean {
        if (!this.activeOptions) return false;
        return Boolean(
            carrier.isGift ||
            carrier.waypointsLooped ||
            this.activeOptions.pinnedCarrierIds?.has(carrier._id.toString()),
        );
    }

    private _canWarpBetween(
        game: Game,
        player: Player,
        a: Star,
        b: Star,
    ): boolean {
        if (!a.warpGate || !b.warpGate) return false;
        if (!a.ownedByPlayerId || !b.ownedByPlayerId) return false;
        if (this.activeOptions?.diplomacy) {
            return (
                this._isAllied(game, player, a.ownedByPlayerId) &&
                this._isAllied(game, player, b.ownedByPlayerId)
            );
        }
        const pid = player._id.toString();
        const aOk =
            a.ownedByPlayerId.toString() === pid ||
            !this._isEnemy(game, player, a.ownedByPlayerId);
        const bOk =
            b.ownedByPlayerId.toString() === pid ||
            !this._isEnemy(game, player, b.ownedByPlayerId);
        return aOk && bOk;
    }

    private _hopTicks(
        game: Game,
        player: Player,
        distMatrix: Float64Array,
        S: number,
        i: number,
        j: number,
    ): number {
        const stars = game.galaxy.stars;
        if (
            this.container.starDataService.isStarPairWormHole(
                stars[i],
                stars[j],
            )
        ) {
            return 1; // Wormholes arrive on the next movement tick
        }
        const d = distMatrix[i * S + j];
        const baseSpeed = game.settings.specialGalaxy.carrierSpeed;
        const warp = this._canWarpBetween(game, player, stars[i], stars[j]);
        const speed = warp
            ? baseSpeed * (game.constants.distances.warpSpeedMultiplier || 3)
            : baseSpeed;
        return Math.max(1, Math.ceil(d / speed));
    }

    // Shortest path in travel ticks from every star to every star along
    // traversable stars (own + optionally unclaimed), respecting hyperspace
    // range, wormholes, and warp gates.
    private _shortestPathsFromOwn(
        game: Game,
        player: Player,
        distMatrix: Float64Array,
        hyperspaceDist: number,
        allowUnownedIntermediate: boolean,
    ): { ticks: Float32Array; nextHop: Int16Array } {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        const allowAllyTransit = Boolean(
            !allowUnownedIntermediate &&
            this.activeOptions?.diplomacy?.allowAlliedStarTransit,
        );
        const canTraverse = new Uint8Array(S);
        for (let i = 0; i < S; i++) {
            const owner = stars[i].ownedByPlayerId?.toString();
            if (
                owner === pid ||
                (allowUnownedIntermediate &&
                    !owner &&
                    !this.activeContestedStars?.has(i)) ||
                (allowAllyTransit &&
                    owner != null &&
                    this.activeOptions!.diplomacy!.isAllied(owner))
            ) {
                canTraverse[i] = 1;
            }
        }

        // All-pairs over S=200 sparse hyperspace graph using per-source Dijkstra.
        const ticks = new Float32Array(S * S).fill(Infinity);
        const nextHop = new Int16Array(S * S).fill(-1);
        const adj: number[][] = Array.from({ length: S }, () => []);
        const edgeCost: number[][] = Array.from({ length: S }, () => []);

        for (let i = 0; i < S; i++) {
            for (let j = 0; j < S; j++) {
                if (i === j) continue;
                if (this.activeBlockedEdges?.has(`${i}->${j}`)) continue;
                if (distMatrix[i * S + j] <= hyperspaceDist) {
                    adj[i].push(j);
                    edgeCost[i].push(
                        this._hopTicks(game, player, distMatrix, S, i, j),
                    );
                }
            }
        }

        const visited = new Uint8Array(S);
        for (let src = 0; src < S; src++) {
            if (stars[src].ownedByPlayerId?.toString() !== pid) continue;
            visited.fill(0);
            const base = src * S;
            ticks[base + src] = 0;

            for (let step = 0; step < S; step++) {
                let u = -1;
                let best = Infinity;
                for (let v = 0; v < S; v++) {
                    if (!visited[v] && ticks[base + v] < best) {
                        best = ticks[base + v];
                        u = v;
                    }
                }
                if (u < 0) break;
                visited[u] = 1;
                if (u !== src && !canTraverse[u]) continue;

                const neighbors = adj[u];
                const costs = edgeCost[u];
                for (let k = 0; k < neighbors.length; k++) {
                    const v = neighbors[k];
                    const nd = best + costs[k];
                    if (nd < ticks[base + v]) {
                        ticks[base + v] = nd;
                        nextHop[base + v] = u === src ? v : nextHop[base + u];
                    }
                }
            }
        }

        return { ticks, nextHop };
    }

    private _carrierCost(game: Game): number {
        const mult =
            game.constants.star.infrastructureExpenseMultipliers[
                game.settings.specialGalaxy.carrierCost
            ];
        return this.container.infrastructureCostService.calculateCarrierCost(
            game,
            mult,
        );
    }

    private _canBuildCarrier(
        game: Game,
        player: Player,
        star: Star,
        ownCarrierCount: number,
        reserveCredits: number,
    ): boolean {
        if (this.container.starDataService.isDeadStar(star)) return false;
        if (Math.floor(star.shipsActual ?? star.ships ?? 0) < 1) return false;
        const limit = this.container.carrierService.getCarrierLimit(
            game,
            player,
        );
        if (ownCarrierCount + 1 > limit) return false;
        const cost = this._carrierCost(game);
        return (player.credits ?? 0) - reserveCredits >= cost;
    }

    async executeTick(
        game: Game,
        player: Player,
        decision: AdmiralDecision,
        believedIndustry?: (starIndex: number) => number | undefined,
        starVisibility?: Uint8Array | null,
        options?: TacticalExecutionOptions,
    ) {
        if (player.defeated) return;
        const prevOptions = this.activeOptions;
        const prevBlocked = this.activeBlockedEdges;
        const prevContested = this.activeContestedStars;
        this.activeOptions = options;

        try {
            const pid = player._id.toString();
            const stars = game.galaxy.stars;
            const S = stars.length;
            const starIndexById = new Map<string, number>();
            const ownStarIndices: number[] = [];

            for (let i = 0; i < S; i++) {
                starIndexById.set(stars[i]._id.toString(), i);
                if (stars[i].ownedByPlayerId?.toString() === pid) {
                    ownStarIndices.push(i);
                }
            }
            if (!ownStarIndices.length) return;

            if (options?.diplomacy) {
                const dip = options.diplomacy;
                const blocked = new Set<string>();
                const contested = new Set<number>();
                const c2cEnabled =
                    game.settings.specialGalaxy.carrierToCarrierCombat ===
                    "enabled";

                for (const c of game.galaxy.carriers) {
                    if (
                        c.orbiting != null ||
                        !c.waypoints?.length ||
                        !c.ownedByPlayerId
                    ) {
                        continue;
                    }
                    const ownerStr = c.ownedByPlayerId.toString();
                    if (ownerStr === pid) continue;
                    const wp0 = c.waypoints[0];
                    const srcIdx = starIndexById.get(wp0.source.toString());
                    const dstIdx = starIndexById.get(
                        wp0.destination.toString(),
                    );

                    if (
                        c2cEnabled &&
                        dip.isProtected(ownerStr) &&
                        srcIdx !== undefined &&
                        dstIdx !== undefined
                    ) {
                        blocked.add(`${dstIdx}->${srcIdx}`);
                    }
                    if (
                        !dip.allowContestedNeutralClaims &&
                        dstIdx !== undefined &&
                        !stars[dstIdx].ownedByPlayerId &&
                        (dip.isAllied(ownerStr) || dip.isProtected(ownerStr))
                    ) {
                        contested.add(dstIdx);
                    }
                }
                this.activeBlockedEdges = blocked;
                this.activeContestedStars = contested;
            } else {
                this.activeBlockedEdges = undefined;
                this.activeContestedStars = undefined;
            }

            const prodTicks = game.settings.galaxy.productionTicks;
            const isFirstTick = game.state.tick % prodTicks === 1;
            const isLastTick = game.state.tick % prodTicks === prodTicks - 1;

            // 1. Pool ships on idle orbiting own carriers back onto their stars so
            //    garrison/attack/defense allocation sees one unified ship pool.
            //    Also track active outbound AND returning logistics ferries so
            //    interior stars wait for their returning ferry instead of spamming
            //    new $25 carriers every cycle.
            const idleCarriersAt = new Map<number, Carrier[]>();
            let ownCarrierCount = 0;
            const committedTargets = new Map<number, number>(); // starIdx -> own ships en route
            const activeLogisticsRoutes = new Set<string>();
            const busyLogisticsStars = new Set<number>();

            for (const c of game.galaxy.carriers) {
                if (c.ownedByPlayerId?.toString() !== pid) continue;
                ownCarrierCount++;
                const pinned = this._isCarrierPinned(c);

                if (c.orbiting && (!c.waypoints || c.waypoints.length === 0)) {
                    if (pinned) continue;
                    const idx = starIndexById.get(c.orbiting.toString());
                    if (
                        idx !== undefined &&
                        stars[idx].ownedByPlayerId?.toString() === pid
                    ) {
                        if ((c.ships ?? 0) > 1) {
                            const add = c.ships! - 1;
                            c.ships = 1;
                            stars[idx].shipsActual =
                                (stars[idx].shipsActual ??
                                    stars[idx].ships ??
                                    0) + add;
                            stars[idx].ships = Math.floor(
                                stars[idx].shipsActual!,
                            );
                        }
                        let list = idleCarriersAt.get(idx);
                        if (!list) {
                            list = [];
                            idleCarriersAt.set(idx, list);
                        }
                        list.push(c);
                    }
                } else if (c.waypoints && c.waypoints.length > 0) {
                    if (c.isGift) continue;
                    const w0 = c.waypoints[0];
                    const wLast = c.waypoints[c.waypoints.length - 1];
                    const srcIdx = starIndexById.get(w0.source.toString());
                    const destIdx = starIndexById.get(
                        w0.destination.toString(),
                    );
                    const lastDestIdx = starIndexById.get(
                        wLast.destination.toString(),
                    );
                    const cShips = c.ships ?? 0;

                    if (destIdx !== undefined) {
                        committedTargets.set(
                            destIdx,
                            (committedTargets.get(destIdx) ?? 0) + cShips,
                        );
                    }
                    if (
                        !pinned &&
                        srcIdx !== undefined &&
                        destIdx !== undefined
                    ) {
                        activeLogisticsRoutes.add(`${srcIdx}->${destIdx}`);
                        activeLogisticsRoutes.add(`${destIdx}->${srcIdx}`);
                    }
                    if (
                        !pinned &&
                        lastDestIdx !== undefined &&
                        stars[lastDestIdx].ownedByPlayerId?.toString() === pid
                    ) {
                        busyLogisticsStars.add(lastDestIdx);
                    }

                    // Track multi-hop claim chains, dropAll sinks, collectAll sweeps,
                    // and multi-hop enemy invasion endpoints.
                    for (let wi = 0; wi < c.waypoints.length; wi++) {
                        const wp = c.waypoints[wi];
                        const dIdx = starIndexById.get(
                            wp.destination.toString(),
                        );
                        if (dIdx === undefined) continue;
                        if (!pinned && wp.action === "collectAll") {
                            busyLogisticsStars.add(dIdx);
                        }
                        if (dIdx !== destIdx) {
                            if (
                                !stars[dIdx].ownedByPlayerId ||
                                wp.action === "dropAll" ||
                                (wi === c.waypoints.length - 1 &&
                                    this._isEnemy(
                                        game,
                                        player,
                                        stars[dIdx].ownedByPlayerId,
                                    ))
                            ) {
                                committedTargets.set(
                                    dIdx,
                                    (committedTargets.get(dIdx) ?? 0) +
                                        Math.max(1, cShips),
                                );
                            }
                        }
                    }
                }
            }

            const distMatrix = this._pairDistances(game);
            const hsDist = this.container.distanceService.getHyperspaceDistance(
                game,
                player.research.hyperspace.level,
            );
            let ownGraph = this._shortestPathsFromOwn(
                game,
                player,
                distMatrix,
                hsDist,
                false,
            );
            let claimGraph = this._shortestPathsFromOwn(
                game,
                player,
                distMatrix,
                hsDist,
                true,
            );

            // Compute per-opponent hyperspace jump reach so border threat & Economy
            // safety reflect which stars each enemy can ACTUALLY jump to (rather
            // than marking our deep interior unsafe when our own hyperspace is high).
            const enemyRangeByPid = new Map<string, number>();
            for (const p of game.galaxy.players) {
                if (!p.defeated && p._id.toString() !== pid) {
                    enemyRangeByPid.set(
                        p._id.toString(),
                        this.container.distanceService.getHyperspaceDistance(
                            game,
                            p.research.hyperspace.level,
                        ),
                    );
                }
            }
            const hostileBorder = new Uint8Array(S);
            const nearbyEnemyShipsAt = new Float32Array(S);
            const baselineGarrison = new Float32Array(S);
            const dip = options?.diplomacy;
            const deterNeutral =
                dip != null && dip.neutralBorderPosture !== "open";
            const watchAllied =
                dip != null && dip.alliedBorderPosture === "watch";

            for (const i of ownStarIndices) {
                let nearbyEnemyShips = 0;
                let nearbyDeterShips = 0;
                for (let j = 0; j < S; j++) {
                    const ownerId = stars[j].ownedByPlayerId;
                    if (!ownerId) continue;
                    const ownerStr = ownerId.toString();
                    if (ownerStr === pid) continue;
                    const eRange = enemyRangeByPid.get(ownerStr) ?? 0;
                    if (distMatrix[i * S + j] > eRange) continue;

                    if (this._isEnemy(game, player, ownerId)) {
                        hostileBorder[i] = 1;
                        nearbyEnemyShips += stars[j].ships ?? 0;
                    } else if (
                        (deterNeutral && dip!.isProtected(ownerStr)) ||
                        (watchAllied && dip!.isAllied(ownerStr))
                    ) {
                        nearbyDeterShips += stars[j].ships ?? 0;
                    }
                }
                nearbyEnemyShipsAt[i] = nearbyEnemyShips;
                if (hostileBorder[i]) {
                    baselineGarrison[i] = Math.min(
                        40,
                        Math.ceil(nearbyEnemyShips * 0.2),
                    );
                }
                if (nearbyDeterShips > 0) {
                    baselineGarrison[i] = Math.max(
                        baselineGarrison[i],
                        Math.min(20, Math.ceil(nearbyDeterShips * 0.12)),
                    );
                }
            }

            const garrisonTarget = new Float32Array(S);
            for (const i of ownStarIndices) {
                garrisonTarget[i] =
                    decision.garrisonTarget?.[i] != null
                        ? Math.max(0, decision.garrisonTarget[i])
                        : baselineGarrison[i];
            }

            // Budget & reserve setup: [science, industry, economy, warpGate, save]
            const rawSplit = decision.budgetSplit ?? [
                0.2, 0.28, 0.45, 0.04, 0.03,
            ];
            const splitSum =
                rawSplit.reduce((a, b) => a + Math.max(0, b), 0) || 1.0;
            const normSplit = rawSplit.map((x) => Math.max(0, x) / splitSum);
            while (normSplit.length < 5) normSplit.push(0);

            const maxAttackTicks = decision.maxAttackTicks ?? prodTicks * 2;
            const protectedCash = Math.max(
                0,
                Math.floor(
                    (options?.cashReserveCredits ?? 0) +
                        (options?.scheduledActionCredits ?? 0),
                ),
            );
            const scheduledOnlyCash = Math.max(
                0,
                Math.floor(options?.scheduledActionCredits ?? 0),
            );
            const allowEmergencyDip =
                options != null &&
                options.allowEmergencyDefenseReserveDip !== false;
            let emergencyDipUsed = false;

            // 2. Detect incoming enemy waves, triage doomed stars vs holdable stars,
            //    reinforce holdable stars, and evacuate doomed stars just before impact.
            const doomedStars = new Set<number>();
            const lockedShips = new Float32Array(S);

            const incomingWaves = this._findIncomingAttacks(
                game,
                player,
                starIndexById,
            );
            const triageThreshold = decision.triageThreshold ?? 0.85;

            for (const wave of incomingWaves) {
                const defIdx = wave.starIndex;
                const defStar = wave.star;
                let neededExtra = this._shipsNeededToHold(
                    game,
                    player,
                    defStar,
                    defIdx,
                    wave.ticksUntil,
                    wave.carriers,
                    committedTargets.get(defIdx) ?? 0,
                );

                if (neededExtra <= 0) {
                    lockedShips[defIdx] = Math.max(
                        lockedShips[defIdx],
                        defStar.ships ?? 0,
                    );
                    continue;
                }

                const defReserveFloor =
                    allowEmergencyDip && !emergencyDipUsed
                        ? scheduledOnlyCash
                        : protectedCash;

                const helpers: { src: number; eta: number; avail: number }[] =
                    [];
                let totalReachable = 0;
                for (const u of ownStarIndices) {
                    if (u === defIdx || doomedStars.has(u)) continue;
                    const eta = ownGraph.ticks[u * S + defIdx];
                    if (eta <= wave.ticksUntil) {
                        const free = this._availableShipsAt(
                            stars[u],
                            idleCarriersAt.get(u),
                            lockedShips[u],
                        );
                        const canCarry =
                            (idleCarriersAt.get(u)?.length ?? 0) > 0 ||
                            this._canBuildCarrier(
                                game,
                                player,
                                stars[u],
                                ownCarrierCount,
                                defReserveFloor,
                            );
                        if (free > 1 && canCarry) {
                            helpers.push({ src: u, eta, avail: free });
                            totalReachable += free;
                        }
                    }
                }

                if (
                    totalReachable < neededExtra * triageThreshold &&
                    options != null &&
                    options.allowEmergencySpecialistDip !== false
                ) {
                    const swungNeeded =
                        await this._tryDefensiveStarSpecialistSwing(
                            game,
                            player,
                            defStar,
                            defIdx,
                            wave,
                            committedTargets.get(defIdx) ?? 0,
                            totalReachable,
                        );
                    if (swungNeeded != null) {
                        neededExtra = swungNeeded;
                    }
                }

                if (totalReachable < neededExtra * triageThreshold) {
                    doomedStars.add(defIdx);
                    garrisonTarget[defIdx] = 0;
                    const standAndBleed = this._shouldStandAndBleed(
                        game,
                        player,
                        wave,
                        ownStarIndices,
                        doomedStars,
                        distMatrix,
                        hsDist,
                        idleCarriersAt,
                        lockedShips,
                    );
                    if (standAndBleed) {
                        lockedShips[defIdx] = Math.max(
                            lockedShips[defIdx],
                            defStar.ships ?? 0,
                        );
                    } else if (wave.ticksUntil <= 3) {
                        ownCarrierCount = await this._evacuateStar(
                            game,
                            player,
                            defIdx,
                            ownStarIndices,
                            doomedStars,
                            ownGraph,
                            idleCarriersAt,
                            ownCarrierCount,
                            protectedCash,
                        );
                        lockedShips[defIdx] = Math.max(lockedShips[defIdx], 1);
                    }
                    continue;
                }

                lockedShips[defIdx] = Math.max(
                    lockedShips[defIdx],
                    defStar.ships ?? 0,
                );
                helpers.sort((a, b) => a.eta - b.eta);
                let remainingNeed = neededExtra;
                for (const h of helpers) {
                    if (remainingNeed <= 0) break;
                    const send = Math.min(remainingNeed, h.avail);
                    if (send <= 0) continue;

                    if (wave.ticksUntil - h.eta > 1) {
                        lockedShips[h.src] += send;
                        remainingNeed -= send;
                    } else {
                        const waypoints = this._buildPathWaypoints(
                            game,
                            h.src,
                            defIdx,
                            ownGraph.nextHop,
                            "dropAll",
                            true,
                        );
                        const hadIdle =
                            (idleCarriersAt.get(h.src)?.length ?? 0) > 0;
                        let launchReserve = protectedCash;
                        let dippingNow = false;
                        if (
                            !hadIdle &&
                            allowEmergencyDip &&
                            !emergencyDipUsed &&
                            !this._canBuildCarrier(
                                game,
                                player,
                                stars[h.src],
                                ownCarrierCount,
                                protectedCash,
                            ) &&
                            this._canBuildCarrier(
                                game,
                                player,
                                stars[h.src],
                                ownCarrierCount,
                                scheduledOnlyCash,
                            )
                        ) {
                            launchReserve = scheduledOnlyCash;
                            dippingNow = true;
                        }
                        const launched = await this._launchShips(
                            game,
                            player,
                            h.src,
                            send,
                            waypoints,
                            idleCarriersAt,
                            ownCarrierCount,
                            launchReserve,
                        );
                        if (launched.launched) {
                            if (dippingNow) {
                                emergencyDipUsed = true;
                                options?.onAlert?.({
                                    kind: "emergency_reserve_dip",
                                    starIndex: defIdx,
                                    starId: defStar._id.toString(),
                                    starName: defStar.name,
                                    ships: launched.shipsSent,
                                    ticksUntil: wave.ticksUntil,
                                    message: `Emergency defense carrier purchased at ${stars[h.src].name} dipping into cash reserve to defend ${defStar.name}.`,
                                });
                            }
                            ownCarrierCount = launched.ownCarrierCount;
                            remainingNeed -= launched.shipsSent;
                            committedTargets.set(
                                defIdx,
                                (committedTargets.get(defIdx) ?? 0) +
                                    launched.shipsSent,
                            );
                        }
                    }
                }
            }

            if (
                options != null &&
                options.allowEmergencySpecialistDip !== false
            ) {
                await this._tryOffensiveCarrierSpecialistSwing(
                    game,
                    player,
                    starIndexById,
                    distMatrix,
                    hsDist,
                );
            }

            // Shield `protectedCash` (`cashReserveCredits + scheduledActionCredits`)
            // across Steps 3..6 so Admiral never spends reserved credits on upgrades,
            // claims, invasions, or logistics.
            const heldCash =
                protectedCash > 0
                    ? Math.min(Math.max(0, player.credits ?? 0), protectedCash)
                    : 0;
            if (heldCash > 0) {
                player.credits = (player.credits ?? 0) - heldCash;
            }

            try {
                // 3. Targeted cycle infrastructure & Warp Gate spending FIRST (skipping
                //    doomed stars) right when cycle payouts arrive (`isFirstTick` for
                //    Science/Industry/WarpGates, `isLastTick` for Economy compounding),
                //    while reserving carrier cash for immediate reachable neutral claims.
                if (isFirstTick || isLastTick) {
                    let reachableUnclaimed = 0;
                    let totalIdleCarriers = 0;
                    for (const u of ownStarIndices) {
                        totalIdleCarriers += idleCarriersAt.get(u)?.length ?? 0;
                    }
                    if (isFirstTick) {
                        const sds = this.container.starDataService;
                        for (let t = 0; t < S; t++) {
                            if (
                                stars[t].ownedByPlayerId ||
                                sds.isDeadStar(stars[t]) ||
                                this.activeContestedStars?.has(t) ||
                                (committedTargets.get(t) ?? 0) > 0
                            ) {
                                continue;
                            }
                            for (const u of ownStarIndices) {
                                if (
                                    claimGraph.ticks[u * S + t] <=
                                    maxAttackTicks
                                ) {
                                    reachableUnclaimed++;
                                    break;
                                }
                            }
                        }
                    }
                    const claimReserve =
                        Math.min(
                            6,
                            Math.max(0, reachableUnclaimed - totalIdleCarriers),
                        ) * this._carrierCost(game);

                    const builtWarpGates = this._spendCycleCredits(
                        game,
                        player,
                        ownStarIndices,
                        doomedStars,
                        hostileBorder,
                        decision,
                        normSplit,
                        claimReserve,
                        isFirstTick,
                        isLastTick,
                        ownGraph,
                        distMatrix,
                        hsDist,
                        starIndexById,
                    );
                    if (builtWarpGates > 0) {
                        ownGraph = this._shortestPathsFromOwn(
                            game,
                            player,
                            distMatrix,
                            hsDist,
                            false,
                        );
                        claimGraph = this._shortestPathsFromOwn(
                            game,
                            player,
                            distMatrix,
                            hsDist,
                            true,
                        );
                        await this._rerouteCarriersAfterWarpGate(
                            game,
                            player,
                            starIndexById,
                            distMatrix,
                            ownGraph,
                        );
                    }
                }

                // Policy-driven Economy hold:
                // - On tick % prodTicks === 0 (just after payout, 1 tick before isFirstTick),
                //   hold the non-carrier-cash share `(1 - normSplit[4])` for 1 tick.
                // - During ticks 1..18, hold the exact `ecoTargetHold` recorded on
                //   `isFirstTick` from `normSplit[2]` so `normSplit[4]` (`carrierCash`)
                //   is 100% liquid for carrier surges whenever the policy wants them!
                const upgradeReserve =
                    game.state.tick % prodTicks === 0
                        ? Math.floor(
                              (player.credits ?? 0) *
                                  Math.min(
                                      0.95,
                                      Math.max(0, 1.0 - normSplit[4]),
                                  ),
                          )
                        : Math.min(
                              player.credits ?? 0,
                              this.cycleEcoHold.get(player) ??
                                  Math.floor(
                                      (player.credits ?? 0) * normSplit[2],
                                  ),
                          );
                const claimCarrierReserve =
                    game.state.tick <= prodTicks ? 0 : upgradeReserve;

                // 4. Claim neutral stars (Cycle 0 uses $0 reserve for fast initial
                //    land-grab; later cycles respect the policy's `cycleEcoHold`).
                ownCarrierCount = await this._executeClaims(
                    game,
                    player,
                    ownStarIndices,
                    doomedStars,
                    lockedShips,
                    garrisonTarget,
                    committedTargets,
                    claimGraph,
                    distMatrix,
                    hsDist,
                    decision,
                    idleCarriersAt,
                    ownCarrierCount,
                    claimCarrierReserve,
                    maxAttackTicks,
                );

                // 5. Hybrid Invasions: Multi-Star Synchronized Time-on-Target (ToT)
                //    strikes + Single-Armada Multi-Hop Warp-Gate strikes.
                ownCarrierCount = await this._executeCoordinatedInvasions(
                    game,
                    player,
                    ownStarIndices,
                    doomedStars,
                    lockedShips,
                    garrisonTarget,
                    committedTargets,
                    ownGraph,
                    distMatrix,
                    hsDist,
                    enemyRangeByPid,
                    decision,
                    believedIndustry,
                    starVisibility,
                    idleCarriersAt,
                    ownCarrierCount,
                    upgradeReserve,
                    maxAttackTicks,
                );

                // 6. Proactive front-line staging & Warp-Gate logistics (with Economy
                //    credit reserve, returning-ferry tracking, and collectAll sweeps).
                ownCarrierCount = await this._executeLogistics(
                    game,
                    player,
                    ownStarIndices,
                    doomedStars,
                    lockedShips,
                    garrisonTarget,
                    hostileBorder,
                    nearbyEnemyShipsAt,
                    committedTargets,
                    activeLogisticsRoutes,
                    busyLogisticsStars,
                    ownGraph,
                    distMatrix,
                    hsDist,
                    enemyRangeByPid,
                    decision,
                    idleCarriersAt,
                    ownCarrierCount,
                    upgradeReserve,
                );
            } finally {
                if (heldCash > 0) {
                    player.credits = (player.credits ?? 0) + heldCash;
                }
            }
        } finally {
            this.activeOptions = prevOptions;
            this.activeBlockedEdges = prevBlocked;
            this.activeContestedStars = prevContested;
        }
    }

    private _availableShipsAt(
        star: Star,
        idleCarriers: Carrier[] | undefined,
        locked: number,
    ): number {
        const garrison = Math.floor(star.shipsActual ?? star.ships ?? 0);
        const carrierShips =
            idleCarriers?.reduce((a, c) => a + (c.ships ?? 0), 0) ?? 0;
        return Math.max(0, garrison + carrierShips - Math.ceil(locked));
    }

    private _findIncomingAttacks(
        game: Game,
        player: Player,
        starIndexById: Map<string, number>,
    ): IncomingWave[] {
        const pid = player._id.toString();
        const dip = this.activeOptions?.diplomacy;
        const grouped = new Map<string, IncomingWave>();
        for (const c of game.galaxy.carriers) {
            if (c.orbiting != null || !c.waypoints?.length) continue;
            if (c.isGift) continue;

            const ownerId = c.ownedByPlayerId;
            if (!ownerId) continue;
            const ownerStr = ownerId.toString();
            if (ownerStr === pid) continue;

            const wp = c.waypoints[0];
            const destIdx = starIndexById.get(wp.destination.toString());
            if (destIdx === undefined) continue;
            const destStar = game.galaxy.stars[destIdx];
            if (destStar.ownedByPlayerId?.toString() !== pid) continue;

            const eta =
                this.container.waypointService.calculateWaypointTicksEta(
                    game,
                    c,
                    wp,
                ) || 1;

            if (dip) {
                if (dip.isAllied(ownerStr)) {
                    continue;
                }
                if (dip.isProtected(ownerStr)) {
                    const attackerAlias = game.galaxy.players.find(
                        (p) => p._id.toString() === ownerStr,
                    )?.alias;
                    const attackerLabel = attackerAlias
                        ? `'${attackerAlias}'`
                        : `player ${ownerStr}`;
                    this.activeOptions?.onAlert?.({
                        kind: "surprise_attack",
                        starIndex: destIdx,
                        starId: destStar._id.toString(),
                        starName: destStar.name,
                        attackerPlayerId: ownerStr,
                        attackerAlias,
                        ships: c.ships ?? 1,
                        ticksUntil: eta,
                        carrierIds: [c._id.toString()],
                        message: `Surprise neutral carrier (${c.ships ?? "?"} ships) from ${attackerLabel} inbound to ${destStar.name} (ETA ${eta}t)! Staging defense without counter-invading.`,
                    });
                } else if (!dip.isHostile(ownerStr)) {
                    continue;
                }
            } else if (!this._isEnemy(game, player, ownerId)) {
                continue;
            }

            const key = `${destIdx}:${eta}`;
            let wave = grouped.get(key);
            if (!wave) {
                wave = {
                    starIndex: destIdx,
                    star: destStar,
                    ticksUntil: eta,
                    carriers: [],
                };
                grouped.set(key, wave);
            }
            wave.carriers.push(c);
        }
        return [...grouped.values()].sort(
            (a, b) => a.ticksUntil - b.ticksUntil,
        );
    }

    private _shipsNeededToHold(
        game: Game,
        player: Player,
        defStar: Star,
        defIdx: number,
        ticksUntil: number,
        attackers: Carrier[],
        friendlyEnRouteShips: number,
    ): number {
        const pid = player._id.toString();
        const orbitingFriendly = game.galaxy.carriers.filter(
            (c) =>
                c.orbiting?.toString() === defStar._id.toString() &&
                c.ownedByPlayerId?.toString() === pid &&
                !(this.activeOptions && c.isGift),
        );
        const techs =
            this.container.technologyService.getStarEffectiveTechnologyLevels(
                game,
                defStar,
                true,
            );
        const produced = Math.floor(
            this.container.shipService.calculateStarShipsByTicks(
                techs.manufacturing,
                defStar.infrastructure?.industry ?? 0,
                ticksUntil,
                game.settings.galaxy.productionTicks,
            ),
        );

        // Temporarily project star ships at arrival for combatService.computeStar.
        const origShips = defStar.ships;
        const origActual = defStar.shipsActual;
        const projected =
            Math.floor(defStar.shipsActual ?? defStar.ships ?? 0) +
            produced +
            friendlyEnRouteShips;
        defStar.ships = projected;
        defStar.shipsActual = projected;

        try {
            const result = this.container.combatService.computeStar(
                game,
                defStar,
                attackers.concat(orbitingFriendly),
            );
            if (!result) return 0;
            const defGroup =
                this.container.combatService.getDefenderDetailed(result);
            if (!defGroup || defGroup.shipsAfter > 0) return 0;
            const totalNeeded = this.container.combatService.estimateNeeded(
                result,
                defGroup,
            );
            return Math.max(0, totalNeeded - defGroup.shipsBefore);
        } finally {
            defStar.ships = origShips;
            defStar.shipsActual = origActual;
        }
    }

    private _shouldStandAndBleed(
        game: Game,
        player: Player,
        wave: IncomingWave,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        distMatrix: Float64Array,
        hsDist: number,
        idleCarriersAt: Map<number, Carrier[]>,
        lockedShips: Float32Array,
    ): boolean {
        const pid = player._id.toString();
        const defIdx = wave.starIndex;
        const defStar = wave.star;
        const S = game.galaxy.stars.length;

        const orbitingFriendly = game.galaxy.carriers.filter(
            (c) =>
                c.orbiting?.toString() === defStar._id.toString() &&
                c.ownedByPlayerId?.toString() === pid &&
                !(this.activeOptions && c.isGift),
        );
        const techs =
            this.container.technologyService.getStarEffectiveTechnologyLevels(
                game,
                defStar,
                true,
            );
        const produced = Math.floor(
            this.container.shipService.calculateStarShipsByTicks(
                techs.manufacturing,
                defStar.infrastructure?.industry ?? 0,
                wave.ticksUntil,
                game.settings.galaxy.productionTicks,
            ),
        );

        const origShips = defStar.ships;
        const origActual = defStar.shipsActual;
        const projected =
            Math.floor(defStar.shipsActual ?? defStar.ships ?? 0) + produced;
        defStar.ships = projected;
        defStar.shipsActual = projected;

        let defBefore = 0;
        let attBefore = 0;
        let attAfter = 0;
        let strikeRatio = 0;
        let maxEnemyWep = 1;

        try {
            const result = this.container.combatService.computeStar(
                game,
                defStar,
                wave.carriers.concat(orbitingFriendly),
            );
            if (!result) return false;
            const groups = result.combatGroups;
            const resById = new Map(result.groups.map((rg) => [rg.id, rg]));
            const dGroupIdx = groups.findIndex((g) => g.isDefender);
            if (dGroupIdx < 0) return false;
            const dGroup = groups[dGroupIdx];
            defBefore =
                resById.get(dGroup.id)?.shipsBefore ?? dGroup.originalShips;
            if (defBefore <= 1) return false;

            let bestAStrike = 1;
            let bestDStrike = 1;
            for (let i = 0; i < groups.length; i++) {
                if (i === dGroupIdx) continue;
                const g = groups[i];
                const rg = resById.get(g.id);
                attBefore += rg ? rg.shipsBefore : g.originalShips;
                attAfter += rg ? rg.shipsAfter : 0;
                const dW = dGroup.attackAgainst.get(i)?.total ?? 1;
                const aW = g.attackAgainst.get(dGroupIdx)?.total ?? 1;
                if (aW >= bestAStrike) {
                    bestAStrike = aW;
                    bestDStrike = dW;
                }
                for (const p of g.players) {
                    if (p.research.weapons.level > maxEnemyWep) {
                        maxEnemyWep = p.research.weapons.level;
                    }
                }
            }
            strikeRatio = bestDStrike / Math.max(1, bestAStrike);
        } finally {
            defStar.ships = origShips;
            defStar.shipsActual = origActual;
        }

        const attKilled = Math.max(0, attBefore - attAfter);
        const tradeRatio = attKilled / Math.max(1, defBefore);
        if (strikeRatio < 1.25 || tradeRatio < 1.25) {
            return false;
        }

        // Only burn the garrison if a nearby non-doomed friendly star can
        // immediately counter-capture the weakened enemy spearhead!
        const defBonusAfterCapture =
            game.settings.specialGalaxy.defenderBonus === "enabled"
                ? 1 + (defStar.isAsteroidField ? 1 : 0)
                : 0;
        const ownWep = Math.max(1, player.research.weapons.level);

        for (const v of ownStarIndices) {
            if (v === defIdx || doomedStars.has(v)) continue;
            if (distMatrix[v * S + defIdx] > hsDist) continue;
            const starV = game.galaxy.stars[v];
            const availV = this._availableShipsAt(
                starV,
                idleCarriersAt.get(v),
                lockedShips[v],
            );
            if (availV <= attAfter) continue;
            const hasCarrierOrCash =
                (idleCarriersAt.get(v)?.length ?? 0) > 0 ||
                (player.credits ?? 0) >= this._carrierCost(game);
            if (!hasCarrierOrCash) continue;

            // Simulate counter-capture against `attAfter` enemy survivors
            let eRem = Math.ceil(attAfter * 1.15);
            let oRem = availV;
            const eTotal = maxEnemyWep + defBonusAfterCapture;
            while (eRem > 0 && oRem > 0) {
                const fromE =
                    eRem < eTotal ? eRem + defBonusAfterCapture : eTotal;
                const fromO = oRem < ownWep ? oRem : ownWep;
                eRem = Math.max(0, eRem - fromO);
                oRem = Math.max(0, oRem - fromE);
            }
            if (eRem === 0 && oRem >= 1) {
                return true;
            }
        }
        return false;
    }

    private async _evacuateStar(
        game: Game,
        player: Player,
        doomedIdx: number,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        reserveCredits = 0,
    ): Promise<number> {
        const S = game.galaxy.stars.length;
        const star = game.galaxy.stars[doomedIdx];
        const idle = idleCarriersAt.get(doomedIdx);
        const hasOrbitingCarrier = (idle?.length ?? 0) > 0;
        const poolWithFirstCarrier =
            Math.floor(star.shipsActual ?? star.ships ?? 0) +
            (idle?.[0]?.ships ?? 0);

        // Rule 1 (Rear Guard): Always leave 1 ship on the doomed star so the
        // attacker never gets a 0-casualty walk-in and takes the `1 + B` hit.
        const evacShips = poolWithFirstCarrier - 1;
        if (evacShips < 1) return ownCarrierCount;

        // Rule 2 ($25 Carrier Gate): Do not spend $25 building a new carrier
        // for tiny stacks (< 10 ships), while evacuating already-orbiting
        // carriers freely (saving both the carrier and its ships for $0).
        if (!hasOrbitingCarrier && evacShips < 10) {
            return ownCarrierCount;
        }

        let bestSafe = -1;
        let bestTicks = Infinity;
        for (const v of ownStarIndices) {
            if (v === doomedIdx || doomedStars.has(v)) continue;
            const t = ownGraph.ticks[doomedIdx * S + v];
            if (t < bestTicks) {
                bestTicks = t;
                bestSafe = v;
            }
        }
        if (bestSafe < 0 || !Number.isFinite(bestTicks)) return ownCarrierCount;

        const hop = ownGraph.nextHop[doomedIdx * S + bestSafe];
        if (hop < 0) return ownCarrierCount;

        const waypoints: CarrierWaypointBase<DBObjectId>[] = [
            {
                source: star._id,
                destination: game.galaxy.stars[hop]._id,
                action: "dropAll",
                actionShips: 0,
                delayTicks: 0,
            },
        ];
        const res = await this._launchShips(
            game,
            player,
            doomedIdx,
            evacShips,
            waypoints,
            idleCarriersAt,
            ownCarrierCount,
            reserveCredits,
        );
        return res.ownCarrierCount;
    }

    private _buildPathWaypoints(
        game: Game,
        srcIdx: number,
        dstIdx: number,
        nextHop: Int16Array,
        arrivalAction: "nothing" | "dropAll" = "nothing",
        returnAfterDrop = false,
        delayFirstHop = 0,
    ): CarrierWaypointBase<DBObjectId>[] {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const path: number[] = [srcIdx];
        let cur = srcIdx;
        let guard = 0;
        while (cur !== dstIdx && guard++ < 28) {
            const nxt = nextHop[cur * S + dstIdx];
            if (nxt < 0) return [];
            path.push(nxt);
            cur = nxt;
        }
        if (cur !== dstIdx || path.length < 2) return [];

        const wps: CarrierWaypointBase<DBObjectId>[] = [];
        for (let i = 1; i < path.length; i++) {
            const isLast = i === path.length - 1;
            wps.push({
                source: stars[path[i - 1]]._id,
                destination: stars[path[i]]._id,
                action: isLast ? arrivalAction : "nothing",
                actionShips: 0,
                delayTicks: i === 1 ? delayFirstHop : 0,
            });
        }
        if (returnAfterDrop && wps.length * 2 <= 30) {
            for (let i = path.length - 1; i >= 1; i--) {
                wps.push({
                    source: stars[path[i]]._id,
                    destination: stars[path[i - 1]]._id,
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 0,
                });
            }
        }
        return wps;
    }

    private async _launchShips(
        game: Game,
        player: Player,
        srcIdx: number,
        shipsToSend: number,
        waypoints: CarrierWaypointBase<DBObjectId>[],
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        reserveCredits: number,
    ): Promise<{
        launched: boolean;
        carrier: Carrier | null;
        shipsSent: number;
        ownCarrierCount: number;
    }> {
        if (!waypoints.length || shipsToSend < 1) {
            return {
                launched: false,
                carrier: null,
                shipsSent: 0,
                ownCarrierCount,
            };
        }
        const star = game.galaxy.stars[srcIdx];
        const idle = idleCarriersAt.get(srcIdx);
        let carrier: Carrier | undefined = idle?.shift();

        if (!carrier) {
            if (
                !this._canBuildCarrier(
                    game,
                    player,
                    star,
                    ownCarrierCount,
                    reserveCredits,
                )
            ) {
                return {
                    launched: false,
                    carrier: null,
                    shipsSent: 0,
                    ownCarrierCount,
                };
            }
            try {
                const built =
                    await this.container.starUpgradeService.buildCarrier(
                        game,
                        player,
                        star._id,
                        1,
                        false,
                        this.container.statisticsService,
                    );
                carrier = this.container.carrierService.getById(
                    game,
                    built.carrier._id,
                );
                ownCarrierCount++;
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
                return {
                    launched: false,
                    carrier: null,
                    shipsSent: 0,
                    ownCarrierCount,
                };
            }
        }

        const totalPool = (carrier.ships ?? 0) + (star.ships ?? 0);
        const actualSend = Math.min(Math.floor(shipsToSend), totalPool);
        if (actualSend < 1) {
            return {
                launched: false,
                carrier: null,
                shipsSent: 0,
                ownCarrierCount,
            };
        }

        const remStar = totalPool - actualSend;
        if (carrier.ships !== actualSend) {
            try {
                await this.container.shipTransferService.transfer(
                    game,
                    player,
                    carrier._id,
                    actualSend,
                    star._id,
                    remStar,
                    false,
                );
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
                return {
                    launched: false,
                    carrier: null,
                    shipsSent: 0,
                    ownCarrierCount,
                };
            }
        }

        try {
            await this.container.saveWaypointsService.saveWaypointsForCarrier(
                game,
                player,
                carrier,
                waypoints,
                false,
                false,
            );
        } catch (e: any) {
            if (!(e instanceof ValidationError)) throw e;
            return {
                launched: false,
                carrier: null,
                shipsSent: 0,
                ownCarrierCount,
            };
        }

        return {
            launched: true,
            carrier,
            shipsSent: actualSend,
            ownCarrierCount,
        };
    }

    private _upgradeCost(
        game: Game,
        star: Star,
        type: InfrastructureType,
    ): number | null {
        const expenseKey = game.settings.player.developmentCost[type];
        if (expenseKey === "none") return null;
        if (this.container.starDataService.isDeadStar(star)) return null;
        const ic = this.container.infrastructureCostService;
        const fn =
            type === "economy"
                ? ic.calculateEconomyCost.bind(ic)
                : type === "industry"
                  ? ic.calculateIndustryCost.bind(ic)
                  : ic.calculateScienceCost.bind(ic);
        return this.container.starUpgradeService._calculateUpgradeInfrastructureCost(
            game,
            star,
            expenseKey,
            type,
            fn,
        );
    }

    private _warpGateCost(game: Game, star: Star): number | null {
        const key = game.settings.specialGalaxy.warpgateCost;
        if (key === "none" || star.warpGate) return null;
        if (this.container.starDataService.isDeadStar(star)) return null;
        const mult = game.constants.star.infrastructureExpenseMultipliers[key];
        const techs =
            this.container.technologyService.getStarEffectiveTechnologyLevels(
                game,
                star,
            );
        const tr = this.container.starService.calculateTerraformedResources(
            star,
            techs.terraforming,
        );
        const avg =
            this.container.infrastructureCostService.calculateAverageTerraformedResources(
                tr,
            );
        return this.container.infrastructureCostService.calculateWarpGateCost(
            game,
            mult,
            avg,
        );
    }

    private _spendPoolOnInfra(
        game: Game,
        player: Player,
        budget: number,
        type: InfrastructureType,
        candidates: number[],
        weights: (idx: number) => number,
    ) {
        const respectIgnore = Boolean(
            this.activeOptions?.respectIgnoreBulkUpgrade,
        );
        let remaining = Math.min(budget, player.credits ?? 0);
        while (remaining > 0) {
            let bestIdx = -1;
            let bestScore = 0;
            let bestCost = Infinity;

            for (const idx of candidates) {
                const starCand = game.galaxy.stars[idx];
                if (
                    respectIgnore &&
                    (starCand.ignoreBulkUpgrade as any)?.[type]
                ) {
                    continue;
                }
                const w = weights(idx);
                if (w <= 0) continue;
                const cost = this._upgradeCost(game, starCand, type);
                if (cost == null || cost <= 0 || cost > remaining) continue;
                const score = w / cost;
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = idx;
                    bestCost = cost;
                }
            }
            if (bestIdx < 0 || bestCost > (player.credits ?? 0)) break;

            const star = game.galaxy.stars[bestIdx];
            star.infrastructure[type] = (star.infrastructure[type] ?? 0) + 1;
            player.credits -= bestCost;
            remaining -= bestCost;
        }
    }

    private _edgeWarpTicksSaved(
        game: Game,
        distMatrix: Float64Array,
        S: number,
        hsDist: number,
        u: number,
        v: number,
    ): number {
        const stars = game.galaxy.stars;
        if (
            this.container.starDataService.isStarPairWormHole(
                stars[u],
                stars[v],
            )
        ) {
            return 0;
        }
        const d = distMatrix[u * S + v];
        if (d > hsDist) return 0;
        const baseSpeed = game.settings.specialGalaxy.carrierSpeed;
        const warpMult = game.constants.distances.warpSpeedMultiplier || 3;
        const tNormal = Math.max(1, Math.ceil(d / baseSpeed));
        const tWarp = Math.max(1, Math.ceil(d / (baseSpeed * warpMult)));
        return Math.max(0, tNormal - tWarp);
    }

    private _computeWarpGateTraffic(
        game: Game,
        player: Player,
        safeOwn: number[],
        hostileBorder: Uint8Array,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        starIndexById: Map<string, number>,
    ): {
        starTraffic: Float64Array;
        edgeTraffic: Map<string, number>;
    } {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        const starTraffic = new Float64Array(S);
        const edgeTraffic = new Map<string, number>();

        const addEdgeTraffic = (u: number, v: number, weight: number) => {
            if (weight <= 0 || u === v) return;
            const saved = this._edgeWarpTicksSaved(
                game,
                distMatrix,
                S,
                hsDist,
                u,
                v,
            );
            if (saved <= 0) return;
            const val = weight * saved;
            starTraffic[u] += val;
            starTraffic[v] += val;
            const key = u < v ? `${u}-${v}` : `${v}-${u}`;
            edgeTraffic.set(key, (edgeTraffic.get(key) ?? 0) + val);
        };

        // 1. Current active carrier waypoints weighted by ship count.
        for (const c of game.galaxy.carriers) {
            if (c.ownedByPlayerId?.toString() !== pid) continue;
            if (!c.waypoints.length) continue;
            const ships = Math.max(1, c.ships ?? 1);
            for (const wp of c.waypoints) {
                const u = starIndexById.get(wp.source.toString());
                const v = starIndexById.get(wp.destination.toString());
                if (u !== undefined && v !== undefined) {
                    addEdgeTraffic(u, v, ships);
                }
            }
        }

        // 2. Projected interior -> front shortest-path ship flow per cycle.
        const frontSinks: number[] = [];
        for (const u of safeOwn) {
            if (hostileBorder[u]) frontSinks.push(u);
        }
        if (frontSinks.length === 0 && safeOwn.length >= 2) {
            const scored = safeOwn
                .map((u) => {
                    let minE = Infinity;
                    for (let t = 0; t < S; t++) {
                        if (
                            !this._isEnemy(
                                game,
                                player,
                                stars[t].ownedByPlayerId,
                            )
                        ) {
                            continue;
                        }
                        const d = distMatrix[u * S + t];
                        if (d < minE) minE = d;
                    }
                    return { u, minE };
                })
                .filter((x) => Number.isFinite(x.minE))
                .sort((a, b) => a.minE - b.minE);
            const k = Math.max(1, Math.round(safeOwn.length * 0.25));
            for (let idx = 0; idx < k && idx < scored.length; idx++) {
                frontSinks.push(scored[idx].u);
            }
        }

        const frontSet = new Set(frontSinks);
        for (const u of safeOwn) {
            if (frontSet.has(u)) continue;
            const prod = this._starShipsPerCycle(game, stars[u]);
            if (prod <= 0) continue;
            let bestF = -1;
            let bestT = Infinity;
            for (const f of frontSinks) {
                const t = ownGraph.ticks[u * S + f];
                if (t < bestT) {
                    bestT = t;
                    bestF = f;
                }
            }
            if (bestF < 0 || !Number.isFinite(bestT)) continue;
            let cur = u;
            let guard = 0;
            while (cur !== bestF && guard++ < 14) {
                const nxt = ownGraph.nextHop[cur * S + bestF];
                if (nxt < 0) break;
                addEdgeTraffic(cur, nxt, prod);
                cur = nxt;
            }
        }

        return { starTraffic, edgeTraffic };
    }

    private async _rerouteCarriersAfterWarpGate(
        game: Game,
        player: Player,
        starIndexById: Map<string, number>,
        distMatrix: Float64Array,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        let rerouted = 0;

        for (const c of game.galaxy.carriers) {
            if (c.ownedByPlayerId?.toString() !== pid) continue;
            if (this._isCarrierPinned(c) || c.isGift || c.waypointsLooped) {
                continue;
            }
            if (c.waypoints.length < 2) continue;
            if (c.waypoints.some((w) => (w.delayTicks ?? 0) > 0)) continue;

            const startIdx = c.orbiting == null ? 1 : 0;
            if (c.waypoints.length - startIdx < 2) continue;

            const anchorId =
                startIdx === 1
                    ? c.waypoints[0].destination.toString()
                    : c.orbiting!.toString();
            const anchorIdx = starIndexById.get(anchorId);
            if (anchorIdx === undefined) continue;

            // Find the first `dropAll` (or final own-star return leg) reachable via
            // pure transit (`action === "nothing"`) so we never drop `collectAll` stops.
            let targetWpIdx = -1;
            for (let wi = startIdx; wi < c.waypoints.length; wi++) {
                const act = c.waypoints[wi].action;
                if (act === "dropAll") {
                    targetWpIdx = wi;
                    break;
                }
                if (act !== "nothing") {
                    break;
                }
                if (wi === c.waypoints.length - 1) {
                    targetWpIdx = wi;
                }
            }
            if (targetWpIdx < startIdx + 1) continue;

            const targetStarIdx = starIndexById.get(
                c.waypoints[targetWpIdx].destination.toString(),
            );
            if (
                targetStarIdx === undefined ||
                stars[targetStarIdx].ownedByPlayerId?.toString() !== pid
            ) {
                continue;
            }

            // Compute current tick cost across `startIdx .. targetWpIdx`.
            let oldTicks = 0;
            let validOld = true;
            for (let wi = startIdx; wi <= targetWpIdx; wi++) {
                const u = starIndexById.get(c.waypoints[wi].source.toString());
                const v = starIndexById.get(
                    c.waypoints[wi].destination.toString(),
                );
                if (u === undefined || v === undefined) {
                    validOld = false;
                    break;
                }
                oldTicks += this._hopTicks(game, player, distMatrix, S, u, v);
            }
            if (!validOld) continue;

            const newTicks = ownGraph.ticks[anchorIdx * S + targetStarIdx];
            if (!Number.isFinite(newTicks) || newTicks >= oldTicks) continue;

            const targetWp = c.waypoints[targetWpIdx];
            const finalAction: "dropAll" | "nothing" =
                targetWp.action === "dropAll" ? "dropAll" : "nothing";
            const rebuiltSegment = this._buildPathWaypoints(
                game,
                anchorIdx,
                targetStarIdx,
                ownGraph.nextHop,
                finalAction,
                false,
                0,
            );
            if (!rebuiltSegment.length) continue;

            const newWaypoints: CarrierWaypointBase<DBObjectId>[] = [
                ...c.waypoints.slice(0, startIdx).map((w) => ({
                    source: w.source,
                    destination: w.destination,
                    action: w.action,
                    actionShips: w.actionShips,
                    delayTicks: w.delayTicks,
                })),
                ...rebuiltSegment,
                ...c.waypoints.slice(targetWpIdx + 1).map((w) => ({
                    source: w.source,
                    destination: w.destination,
                    action: w.action,
                    actionShips: w.actionShips,
                    delayTicks: w.delayTicks,
                })),
            ];
            if (newWaypoints.length > 30) continue;

            try {
                await this.container.saveWaypointsService.saveWaypointsForCarrier(
                    game,
                    player,
                    c,
                    newWaypoints,
                    false,
                    false,
                );
                rerouted++;
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
            }
        }

        return rerouted;
    }

    private _spendCycleCredits(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        hostileBorder: Uint8Array,
        decision: AdmiralDecision,
        normSplit: number[],
        claimReserve: number,
        isFirstTick: boolean,
        isLastTick: boolean,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        starIndexById: Map<string, number>,
    ): number {
        const safeOwn = ownStarIndices.filter((i) => !doomedStars.has(i));
        if (!safeOwn.length) return 0;
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        let builtWarpGates = 0;

        if (isFirstTick) {
            const totalCredits = player.credits ?? 0;
            this.cycleEcoHold.set(
                player,
                Math.floor(totalCredits * normSplit[2]),
            );
            const maxFirstSpend = Math.max(0, totalCredits - claimReserve);
            const firstShare = normSplit[0] + normSplit[1] + normSplit[3];
            if (maxFirstSpend <= 0 || firstShare <= 0) return 0;

            let sciBudget = Math.floor(totalCredits * normSplit[0]);
            let indBudget = Math.floor(totalCredits * normSplit[1]);
            let wgBudget = Math.floor(totalCredits * normSplit[3]);
            const plannedFirst = sciBudget + indBudget + wgBudget;
            if (plannedFirst > maxFirstSpend && plannedFirst > 0) {
                const scale = maxFirstSpend / plannedFirst;
                sciBudget = Math.floor(sciBudget * scale);
                indBudget = Math.floor(indBudget * scale);
                wgBudget = Math.floor(wgBudget * scale);
            }

            // Science: strongly prefer Research Stations (2x multiplier!),
            // Binary Stars, and safe interior core stars.
            this._spendPoolOnInfra(
                game,
                player,
                sciBudget,
                "science",
                safeOwn,
                (i) => {
                    const base = decision.sciencePriority?.[i] ?? 1.0;
                    const specBoost = stars[i].specialistId === 11 ? 2.2 : 1.0;
                    const binBoost = stars[i].isBinaryStar ? 1.15 : 1.0;
                    const borderPenalty = hostileBorder[i] ? 0.5 : 1.0;
                    return base * specBoost * binBoost * borderPenalty;
                },
            );

            // Industry: prefer Space Docks (2), War Machines (7) & frontier staging hubs.
            this._spendPoolOnInfra(
                game,
                player,
                indBudget,
                "industry",
                safeOwn,
                (i) => {
                    const base = decision.industryPriority?.[i] ?? 1.0;
                    const specBoost =
                        stars[i].specialistId === 2 ||
                        stars[i].specialistId === 7
                            ? 2.2
                            : 1.0;
                    const borderBoost = hostileBorder[i] ? 1.25 : 1.0;
                    return base * specBoost * borderBoost;
                },
            );

            // Warp Gates: weight by actual carrier waypoint + interior->front ship
            // traffic on >=2-tick edges, with a lower threshold & 3x boost for stars
            // within hyperspace range of an existing own or allied Warp Gate.
            let wgRem = Math.min(
                wgBudget,
                Math.max(0, (player.credits ?? 0) - claimReserve),
            );
            if (wgRem > 0 && safeOwn.length >= 2) {
                const { starTraffic, edgeTraffic } =
                    this._computeWarpGateTraffic(
                        game,
                        player,
                        safeOwn,
                        hostileBorder,
                        ownGraph,
                        distMatrix,
                        hsDist,
                        starIndexById,
                    );
                const traversable: number[] = [];
                for (let j = 0; j < S; j++) {
                    const owner = stars[j].ownedByPlayerId;
                    if (!owner) continue;
                    const ok = this.activeOptions?.diplomacy
                        ? this._isAllied(game, player, owner)
                        : owner.toString() === pid ||
                          !this._isEnemy(game, player, owner);
                    if (ok) traversable.push(j);
                }

                while (wgRem > 0) {
                    // Check if we already have an orphaned own Warp Gate that can be paired.
                    let hasUnpairedOrphan = false;
                    for (const gIdx of safeOwn) {
                        if (!stars[gIdx].warpGate) continue;
                        let hasGatedNbr = false;
                        let hasUngatedPair = false;
                        for (const j of traversable) {
                            if (j === gIdx) continue;
                            if (
                                this._edgeWarpTicksSaved(
                                    game,
                                    distMatrix,
                                    S,
                                    hsDist,
                                    gIdx,
                                    j,
                                ) <= 0
                            ) {
                                continue;
                            }
                            if (stars[j].warpGate) hasGatedNbr = true;
                            else if (
                                stars[j].ownedByPlayerId?.toString() === pid
                            )
                                hasUngatedPair = true;
                        }
                        if (!hasGatedNbr && hasUngatedPair) {
                            hasUnpairedOrphan = true;
                            break;
                        }
                    }

                    let bestIdx = -1;
                    let bestScore = 0;
                    let bestCost = Infinity;
                    for (const i of safeOwn) {
                        const cost = this._warpGateCost(game, stars[i]);
                        if (cost == null || cost <= 0 || cost > wgRem) continue;

                        let hasGatedNeighbor = false;
                        let hasPairableNeighbor = false;
                        let connectedTraffic = 0;
                        for (const j of traversable) {
                            if (j === i) continue;
                            if (
                                this._edgeWarpTicksSaved(
                                    game,
                                    distMatrix,
                                    S,
                                    hsDist,
                                    i,
                                    j,
                                ) <= 0
                            ) {
                                continue;
                            }
                            hasPairableNeighbor = true;
                            if (stars[j].warpGate) {
                                hasGatedNeighbor = true;
                                const key = i < j ? `${i}-${j}` : `${j}-${i}`;
                                connectedTraffic += edgeTraffic.get(key) ?? 0;
                            }
                        }
                        if (!hasPairableNeighbor) continue;

                        const ind = stars[i].infrastructure?.industry ?? 0;
                        const traf = starTraffic[i];
                        const prior = decision.warpGatePriority?.[i] ?? 1.0;

                        let w = 0;
                        if (hasGatedNeighbor) {
                            // Lower threshold & 3x boost: pays off immediately!
                            if (traf <= 0 && ind < 2) continue;
                            w =
                                prior *
                                3.0 *
                                (1 +
                                    0.15 * connectedTraffic +
                                    0.08 * traf +
                                    0.2 * ind);
                        } else {
                            // Higher threshold to seed an isolated ("first") Warp Gate.
                            if (traf < 12 && !(traf >= 4 && ind >= 4)) continue;
                            const orphanPenalty = hasUnpairedOrphan
                                ? 0.35
                                : 1.0;
                            w =
                                prior *
                                orphanPenalty *
                                (1 + 0.08 * traf + 0.2 * ind);
                        }

                        const score = w / cost;
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdx = i;
                            bestCost = cost;
                        }
                    }
                    if (bestIdx < 0 || bestCost > (player.credits ?? 0)) break;
                    stars[bestIdx].warpGate = true;
                    player.credits -= bestCost;
                    wgRem -= bestCost;
                    builtWarpGates++;
                }
            }

            // Roll 60% of unspent Warp-Gate budget into Industry now and leave
            // 40% to compound into Economy on `isLastTick`.
            const rollInd = Math.floor(wgRem * 0.6);
            if (rollInd > 0 && (player.credits ?? 0) > claimReserve) {
                this._spendPoolOnInfra(
                    game,
                    player,
                    Math.min(rollInd, (player.credits ?? 0) - claimReserve),
                    "industry",
                    safeOwn,
                    (i) => {
                        const base = decision.industryPriority?.[i] ?? 1.0;
                        const specBoost =
                            stars[i].specialistId === 2 ||
                            stars[i].specialistId === 7
                                ? 2.2
                                : 1.0;
                        return base * specBoost;
                    },
                );
            }
        }

        if (isLastTick) {
            // Right before Tick 20's production payout (1 tick away!), compound
            // remaining credits into Economy on safe interior stars unless the
            // policy explicitly holds a large cross-cycle war chest (`normSplit[4] > 0.05`).
            const saveFrac =
                normSplit[4] <= 0.05
                    ? 0
                    : normSplit[4] /
                      Math.max(1e-6, normSplit[2] + normSplit[4]);
            const lastReserve = Math.floor(
                (player.credits ?? 0) * Math.min(0.5, saveFrac),
            );
            const ecoBudget = Math.max(0, (player.credits ?? 0) - lastReserve);
            const hasInteriorSafe = safeOwn.some((i) => !hostileBorder[i]);

            this._spendPoolOnInfra(
                game,
                player,
                ecoBudget,
                "economy",
                safeOwn,
                (i) => {
                    const base = decision.economyPriority?.[i] ?? 1.0;
                    const borderPenalty =
                        hostileBorder[i] && hasInteriorSafe ? 0.08 : 1.0;
                    return base * borderPenalty;
                },
            );
            this.cycleEcoHold.set(player, 0);
        }

        return builtWarpGates;
    }

    private _starStrategicValue(star: Star): number {
        const nr = star.naturalResources ?? {
            economy: 0,
            industry: 0,
            science: 0,
        };
        const infra = star.infrastructure ?? {
            economy: 0,
            industry: 0,
            science: 0,
        };
        let v =
            10 +
            0.4 * ((nr.economy || 0) + (nr.industry || 0) + (nr.science || 0)) +
            4.0 * (infra.economy || 0) +
            6.0 * (infra.industry || 0) +
            5.0 * (infra.science || 0);
        if (star.isBinaryStar) v *= 1.4;
        if (star.wormHoleToStarId) v *= 1.35;
        if (star.isPulsar) v *= 1.2;
        if (star.isBlackHole) v *= 1.25;
        if (star.warpGate) v += 15;
        return v;
    }

    private async _executeClaims(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        lockedShips: Float32Array,
        garrisonTarget: Float32Array,
        committedTargets: Map<number, number>,
        claimGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        decision: AdmiralDecision,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        carrierReserve: number,
        maxAttackTicks: number,
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        const sds = this.container.starDataService;

        // Precompute other players' owned stars for frontier contestation distance (#3).
        const otherOwned: number[] = [];
        for (let i = 0; i < S; i++) {
            const owner = stars[i].ownedByPlayerId;
            if (owner && owner.toString() !== pid) {
                otherOwned.push(i);
            }
        }

        // Precompute which unclaimed stars are already 1-hop reachable from own stars (#6).
        const withinOwnHs = new Uint8Array(S);
        for (let t = 0; t < S; t++) {
            for (const u of ownStarIndices) {
                if (distMatrix[u * S + t] <= hsDist) {
                    withinOwnHs[t] = 1;
                    break;
                }
            }
        }

        const candidates: { target: number; score: number }[] = [];
        for (let t = 0; t < S; t++) {
            if (stars[t].ownedByPlayerId || sds.isDeadStar(stars[t])) continue;
            if (this.activeContestedStars?.has(t)) continue;
            if ((committedTargets.get(t) ?? 0) > 0) continue;
            const desire = decision.attackDesire?.[t] ?? 1.0;
            if (desire <= 0) continue;

            let bestEta = Infinity;
            let myMinDist = Infinity;
            for (const u of ownStarIndices) {
                if (claimGraph.ticks[u * S + t] < bestEta) {
                    bestEta = claimGraph.ticks[u * S + t];
                }
                const d = distMatrix[u * S + t];
                if (d < myMinDist) myMinDist = d;
            }
            if (bestEta > maxAttackTicks) continue;

            let otherMinDist = Infinity;
            for (const o of otherOwned) {
                const d = distMatrix[o * S + t];
                if (d < otherMinDist) otherMinDist = d;
            }

            // #3: Frontier-first contestation multiplier — stake out contested border
            // stars near neighbors ahead of safe backline stars nobody else can reach.
            let frontierMult = 1.0;
            if (Number.isFinite(otherMinDist) && myMinDist > 0) {
                const ratio = otherMinDist / myMinDist;
                if (ratio <= 1.15) frontierMult = 2.4;
                else if (ratio <= 1.5) frontierMult = 1.75;
                else if (ratio >= 2.2) frontierMult = 0.72;
            }

            // #4: Star-quality weighting — prioritize Binary Stars and high natural-resource
            // stars (stacking with `_starStrategicValue`'s base special-star multipliers).
            const nr = stars[t].naturalResources ?? {
                economy: 0,
                industry: 0,
                science: 0,
            };
            const nrSum =
                (nr.economy || 0) + (nr.industry || 0) + (nr.science || 0);
            let qualityMult = 1.0;
            if (stars[t].isBinaryStar) qualityMult *= 1.45;
            if (nrSum >= 90) qualityMult *= 1.3;
            else if (nrSum >= 70) qualityMult *= 1.12;

            candidates.push({
                target: t,
                score:
                    (this._starStrategicValue(stars[t]) *
                        desire *
                        frontierMult *
                        qualityMult) /
                    Math.pow(Math.max(1, bestEta), 0.8),
            });
        }
        candidates.sort((a, b) => b.score - a.score);

        for (const { target } of candidates) {
            if ((committedTargets.get(target) ?? 0) > 0) continue;
            let bestSrc = -1;
            let bestEta = Infinity;
            for (const u of ownStarIndices) {
                if (doomedStars.has(u)) continue;
                const eta = claimGraph.ticks[u * S + target];
                if (eta > maxAttackTicks || eta >= bestEta) continue;
                // Neutral claims take 1 ship; never block expansion behind passive
                // border garrisonTarget — only hold active incoming-attack locks.
                const surplus = this._availableShipsAt(
                    stars[u],
                    idleCarriersAt.get(u),
                    lockedShips[u],
                );
                const canCarry =
                    (idleCarriersAt.get(u)?.length ?? 0) > 0 ||
                    this._canBuildCarrier(
                        game,
                        player,
                        stars[u],
                        ownCarrierCount,
                        carrierReserve,
                    );
                if (surplus >= 1 && canCarry) {
                    bestEta = eta;
                    bestSrc = u;
                }
            }
            if (bestSrc < 0) continue;

            const wps = this._buildPathWaypoints(
                game,
                bestSrc,
                target,
                claimGraph.nextHop,
                "nothing",
                false,
            );
            if (!wps.length) continue;

            // #6: Multi-ship branching expansion carriers — count unclaimed stars
            // reachable from `target` that aren't already within our existing empire's
            // hyperspace range so `target` can immediately split a 2nd carrier on arrival.
            let unclaimedBeyond = 0;
            for (let k = 0; k < S; k++) {
                if (k === target || withinOwnHs[k]) continue;
                if (stars[k].ownedByPlayerId || sds.isDeadStar(stars[k])) {
                    continue;
                }
                if (distMatrix[target * S + k] <= hsDist) {
                    unclaimedBeyond++;
                }
            }
            const branchShips =
                unclaimedBeyond >= 2 ? Math.min(3, unclaimedBeyond) : 1;

            const surplus = this._availableShipsAt(
                stars[bestSrc],
                idleCarriersAt.get(bestSrc),
                lockedShips[bestSrc],
            );
            const send = Math.min(
                surplus,
                Math.min(4, Math.max(wps.length, branchShips)),
            );
            const res = await this._launchShips(
                game,
                player,
                bestSrc,
                send,
                wps,
                idleCarriersAt,
                ownCarrierCount,
                carrierReserve,
            );
            if (res.launched) {
                ownCarrierCount = res.ownCarrierCount;
                for (const wp of wps) {
                    const idx = stars.findIndex(
                        (s) => s._id.toString() === wp.destination.toString(),
                    );
                    if (idx >= 0) committedTargets.set(idx, res.shipsSent);
                }
            }
        }

        // Optional Explorer specialist outreach for stars beyond normal hyperspace range.
        // Conserves initial 5 specialist tokens for Research Station (id: 11)
        // unless zero normal claims are reachable (stranded) or tokens >= 8.
        if (decision.autoCarrierSpecialist !== false) {
            ownCarrierCount = await this._dispatchExplorers(
                game,
                player,
                ownStarIndices,
                doomedStars,
                committedTargets,
                distMatrix,
                hsDist,
                idleCarriersAt,
                ownCarrierCount,
                candidates.length > 0,
            );
        }

        return ownCarrierCount;
    }

    private async _dispatchExplorers(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        committedTargets: Map<number, number>,
        distMatrix: Float64Array,
        nearDist: number,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        hasNormalCandidates = false,
    ): Promise<number> {
        const c = this.container;
        const pid = player._id.toString();
        const stars = game.galaxy.stars;
        const S = stars.length;
        const farDist = c.distanceService.getHyperspaceDistance(
            game,
            player.research.hyperspace.level + 3,
        );

        const freeIndices: number[] = [];
        for (let i = 0; i < S; i++) {
            if (
                !stars[i].ownedByPlayerId &&
                !c.starDataService.isDeadStar(stars[i]) &&
                !this.activeContestedStars?.has(i) &&
                !(committedTargets.get(i) ?? 0)
            ) {
                freeIndices.push(i);
            }
        }
        if (!freeIndices.length) return ownCarrierCount;

        const explorers = game.galaxy.carriers.filter(
            (car) =>
                car.ownedByPlayerId?.toString() === pid &&
                car.specialistId === SPEC_EXPLORER &&
                !this._isCarrierPinned(car),
        );

        for (const car of explorers) {
            if (!car.orbiting || car.waypoints?.length) continue;
            const fromIdx = stars.findIndex(
                (s) => s._id.toString() === car.orbiting!.toString(),
            );
            if (fromIdx < 0) continue;
            let bestTarget = -1;
            let bestD = Infinity;
            for (const t of freeIndices) {
                if ((committedTargets.get(t) ?? 0) > 0) continue;
                if (this.activeBlockedEdges?.has(`${fromIdx}->${t}`)) continue;
                const d = distMatrix[fromIdx * S + t];
                if (d <= farDist && d < bestD) {
                    bestD = d;
                    bestTarget = t;
                }
            }
            if (bestTarget >= 0) {
                try {
                    await c.saveWaypointsService.saveWaypointsForCarrier(
                        game,
                        player,
                        car,
                        [
                            {
                                source: stars[fromIdx]._id,
                                destination: stars[bestTarget]._id,
                                action: "nothing",
                                actionShips: 0,
                                delayTicks: 0,
                            },
                        ],
                        false,
                        false,
                    );
                    committedTargets.set(bestTarget, car.ships ?? 1);
                    const idle = idleCarriersAt.get(fromIdx);
                    if (idle) {
                        const k = idle.indexOf(car);
                        if (k >= 0) idle.splice(k, 1);
                    }
                } catch (e: any) {
                    if (!(e instanceof ValidationError)) throw e;
                }
            }
        }

        if (explorers.length > 0) return ownCarrierCount;

        // Protect starting 5 specialist tokens for Research Station (id: 11):
        // only buy an Explorer (3 tokens) when genuinely stranded with zero
        // normal reachable neutral stars, OR when specialist tokens >= 8.
        const specCurrency = game.settings.specialGalaxy.specialistsCurrency;
        const curSpecBudget = (player as any)[specCurrency] ?? 0;
        if (hasNormalCandidates && curSpecBudget < 8) {
            return ownCarrierCount;
        }

        // Hire one Explorer if an unclaimed star sits exclusively in (nearDist, farDist].
        let bestPair: { src: number; dst: number; d: number } | null = null;
        for (const t of freeIndices) {
            let minOwnD = Infinity;
            let closestOwn = -1;
            for (const u of ownStarIndices) {
                if (doomedStars.has(u)) continue;
                if (this.activeBlockedEdges?.has(`${u}->${t}`)) continue;
                const d = distMatrix[u * S + t];
                if (d < minOwnD) {
                    minOwnD = d;
                    closestOwn = u;
                }
            }
            if (closestOwn >= 0 && minOwnD > nearDist && minOwnD <= farDist) {
                if (!bestPair || minOwnD < bestPair.d) {
                    bestPair = { src: closestOwn, dst: t, d: minOwnD };
                }
            }
        }
        if (!bestPair) return ownCarrierCount;

        if (!this._canAffordCarrierSpecialist(game, player, SPEC_EXPLORER)) {
            return ownCarrierCount;
        }

        let car = idleCarriersAt
            .get(bestPair.src)
            ?.find((x) => x.specialistId == null);
        if (!car) {
            if (
                !this._canBuildCarrier(
                    game,
                    player,
                    stars[bestPair.src],
                    ownCarrierCount,
                    0,
                )
            ) {
                return ownCarrierCount;
            }
            try {
                const built = await c.starUpgradeService.buildCarrier(
                    game,
                    player,
                    stars[bestPair.src]._id,
                    1,
                    false,
                    c.statisticsService,
                );
                car = c.carrierService.getById(game, built.carrier._id);
                ownCarrierCount++;
            } catch (e: any) {
                if (!(e instanceof ValidationError)) throw e;
                return ownCarrierCount;
            }
        } else {
            const idle = idleCarriersAt.get(bestPair.src)!;
            idle.splice(idle.indexOf(car), 1);
        }

        try {
            await c.specialistHireService.hireCarrierSpecialist(
                game,
                player,
                car._id,
                SPEC_EXPLORER,
                c.statisticsService,
            );
            await c.saveWaypointsService.saveWaypointsForCarrier(
                game,
                player,
                car,
                [
                    {
                        source: stars[bestPair.src]._id,
                        destination: stars[bestPair.dst]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                false,
                false,
            );
            committedTargets.set(bestPair.dst, car.ships ?? 1);
        } catch (e: any) {
            if (!(e instanceof ValidationError)) throw e;
        }

        return ownCarrierCount;
    }

    private _canAffordCarrierSpecialist(
        game: Game,
        player: Player,
        specId: number,
        allowReserveDip = false,
    ): boolean {
        if (game.settings.specialGalaxy.specialistCost === "none") return false;
        if (
            this.container.specialistBanService.isCarrierSpecialistBanned(
                game,
                specId,
            )
        ) {
            return false;
        }
        const spec = this.container.specialistService.getByIdCarrier(specId);
        if (!spec) return false;
        const currency = game.settings.specialGalaxy.specialistsCurrency;
        const cost = this.container.specialistService.getSpecialistActualCost(
            game,
            spec,
        )[currency];
        const balance = (player as any)[currency] ?? 0;
        if (balance < cost) return false;
        if (!allowReserveDip) {
            if (currency === "creditsSpecialists") {
                const reserve = Math.max(
                    0,
                    Math.floor(
                        this.activeOptions?.specialistReserveTokens ?? 0,
                    ),
                );
                if (balance - cost < reserve) return false;
            } else if (currency === "credits") {
                const reserve = Math.max(
                    0,
                    Math.floor(
                        (this.activeOptions?.cashReserveCredits ?? 0) +
                            (this.activeOptions?.scheduledActionCredits ?? 0),
                    ),
                );
                if (balance - cost < reserve) return false;
            }
        }
        return true;
    }

    private async _tryDefensiveStarSpecialistSwing(
        game: Game,
        player: Player,
        defStar: Star,
        defIdx: number,
        wave: IncomingWave,
        committedDefShips: number,
        totalReachable: number,
    ): Promise<number | null> {
        if (game.settings.specialGalaxy.specialistCost === "none") return null;
        if (defStar.specialistId != null) return null;

        const currency = game.settings.specialGalaxy.specialistsCurrency;
        const balance = (player as any)[currency] ?? 0;
        const reserve =
            currency === "creditsSpecialists"
                ? Math.max(
                      0,
                      Math.floor(
                          this.activeOptions?.specialistReserveTokens ?? 0,
                      ),
                  )
                : Math.max(
                      0,
                      Math.floor(
                          (this.activeOptions?.cashReserveCredits ?? 0) +
                              (this.activeOptions?.scheduledActionCredits ?? 0),
                      ),
                  );

        const candidates = this.container.specialistService
            .listStar(game)
            .filter((spec) => {
                if (
                    this.container.specialistBanService.isStarSpecialistBanned(
                        game,
                        spec.id,
                    )
                ) {
                    return false;
                }
                const loc = (spec.modifiers?.local ?? {}) as any;
                const sp = (spec.modifiers?.special ?? {}) as any;
                const hasDefCombat =
                    (loc.weapons ?? 0) > 0 ||
                    (loc.defenderBonus ?? 0) > 0 ||
                    (sp.defenderBonus ?? 0) > 0 ||
                    (sp.weapons ?? 0) > 0 ||
                    (sp.deductEnemyWeapons ?? 0) > 0;
                if (!hasDefCombat) return false;
                const cost =
                    this.container.specialistService.getSpecialistActualCost(
                        game,
                        spec,
                    )[currency];
                return cost > 0 && balance >= cost;
            })
            .map((spec) => ({
                spec,
                cost: this.container.specialistService.getSpecialistActualCost(
                    game,
                    spec,
                )[currency],
            }))
            .sort((a, b) => a.cost - b.cost);

        for (const { spec, cost } of candidates) {
            const prevSpecId = defStar.specialistId;
            defStar.specialistId = spec.id;
            const newNeeded = this._shipsNeededToHold(
                game,
                player,
                defStar,
                defIdx,
                wave.ticksUntil,
                wave.carriers,
                committedDefShips,
            );
            defStar.specialistId = prevSpecId;

            if (newNeeded <= totalReachable) {
                try {
                    await this.container.specialistHireService.hireStarSpecialist(
                        game,
                        player,
                        defStar._id,
                        spec.id,
                        this.container.statisticsService,
                    );
                    const dipped = balance - cost < reserve;
                    if (dipped) {
                        this.activeOptions?.onAlert?.({
                            kind: "emergency_specialist_dip",
                            starIndex: defIdx,
                            starId: defStar._id.toString(),
                            starName: defStar.name,
                            specialistId: spec.id,
                            specialistName: spec.name,
                            ships: defStar.ships ?? 0,
                            ticksUntil: wave.ticksUntil,
                            message: `Emergency specialist dip: hired '${spec.name}' on ${defStar.name} (cost ${cost} ${currency === "creditsSpecialists" ? "token(s)" : "cr"}) to swing defense against inbound wave (ETA ${wave.ticksUntil}t).`,
                        });
                    }
                    return newNeeded;
                } catch (e: any) {
                    if (!(e instanceof ValidationError)) throw e;
                }
            }
        }
        return null;
    }

    private async _tryOffensiveCarrierSpecialistSwing(
        game: Game,
        player: Player,
        starIndexById: Map<string, number>,
        distMatrix: Float64Array,
        hsDist: number,
    ): Promise<void> {
        if (game.settings.specialGalaxy.specialistCost === "none") return;
        const stars = game.galaxy.stars;
        const S = stars.length;
        const pid = player._id.toString();
        const currency = game.settings.specialGalaxy.specialistsCurrency;
        const reserve =
            currency === "creditsSpecialists"
                ? Math.max(
                      0,
                      Math.floor(
                          this.activeOptions?.specialistReserveTokens ?? 0,
                      ),
                  )
                : Math.max(
                      0,
                      Math.floor(
                          (this.activeOptions?.cashReserveCredits ?? 0) +
                              (this.activeOptions?.scheduledActionCredits ?? 0),
                      ),
                  );

        const combatSpecs = this.container.specialistService
            .listCarrier(game)
            .filter((spec) => {
                if (
                    this.container.specialistBanService.isCarrierSpecialistBanned(
                        game,
                        spec.id,
                    )
                ) {
                    return false;
                }
                const loc = (spec.modifiers?.local ?? {}) as any;
                const sp = (spec.modifiers?.special ?? {}) as any;
                return (
                    (loc.weapons ?? 0) > 0 ||
                    (sp.weapons ?? 0) > 0 ||
                    (sp.deductEnemyWeapons ?? 0) > 0 ||
                    (sp.starCaptureRewardMultiplier ?? 0) > 1
                );
            })
            .map((spec) => ({
                spec,
                cost: this.container.specialistService.getSpecialistActualCost(
                    game,
                    spec,
                )[currency],
            }))
            .filter((x) => x.cost > 0)
            .sort((a, b) => a.cost - b.cost);

        if (!combatSpecs.length) return;

        for (const c of game.galaxy.carriers) {
            if (c.ownedByPlayerId?.toString() !== pid) continue;
            if (
                !c.orbiting ||
                c.specialistId != null ||
                c.isGift ||
                this._isCarrierPinned(c)
            ) {
                continue;
            }
            if (!c.waypoints.length) continue;

            const orbStar =
                stars[starIndexById.get(c.orbiting.toString()) ?? -1];
            if (!orbStar || orbStar.ownedByPlayerId?.toString() !== pid) {
                continue;
            }
            const dstIdx = starIndexById.get(
                c.waypoints[0].destination.toString(),
            );
            if (dstIdx === undefined) continue;
            const dstStar = stars[dstIdx];
            if (
                !dstStar.ownedByPlayerId ||
                !this._isEnemy(game, player, dstStar.ownedByPlayerId)
            ) {
                continue;
            }

            // Simulate current outcome at `dstStar` (including any visible enemy
            // reinforcement carriers arriving at `dstStar` within our ETA).
            const eta =
                this.container.waypointService.calculateWaypointTicksEta(
                    game,
                    c,
                    c.waypoints[0],
                ) || 1;
            const arriving = [c];
            for (const oc of game.galaxy.carriers) {
                if (oc === c || !oc.waypoints.length) continue;
                if (
                    oc.waypoints[0].destination.toString() !==
                    dstStar._id.toString()
                ) {
                    continue;
                }
                const oEta =
                    this.container.waypointService.calculateWaypointTicksEta(
                        game,
                        oc,
                        oc.waypoints[0],
                    ) || 1;
                if (oEta <= eta) arriving.push(oc);
            }

            const myWins = (res: any): boolean => {
                const defGroup =
                    this.container.combatService.getDefenderDetailed(res);
                const myGroup = (res.groups ?? []).find((g: any) =>
                    (g.players ?? []).some(
                        (p: any) => p._id?.toString() === pid,
                    ),
                );
                return (
                    (myGroup?.shipsAfter ?? 0) > 0 &&
                    (defGroup?.shipsAfter ?? 0) <= 0
                );
            };

            const baseRes = this.container.combatService.computeStar(
                game,
                dstStar,
                arriving,
            );
            if (!baseRes || myWins(baseRes)) continue; // Already winning!

            const balance = (player as any)[currency] ?? 0;
            const orbIdx = starIndexById.get(c.orbiting.toString())!;
            const hopDist = distMatrix[orbIdx * S + dstIdx];

            for (const { spec, cost } of combatSpecs) {
                if (balance < cost) continue;
                // Check hyperspace range if specialist reduces carrier hyperspace (e.g., Lieutenant -1 HS).
                const hsMod = Number(
                    (spec.modifiers?.local as any)?.hyperspace ?? 0,
                );
                if (hsMod < 0) {
                    const effHsLvl = Math.max(
                        1,
                        (player.research.hyperspace.level ?? 1) + hsMod,
                    );
                    const effHsDist =
                        this.container.distanceService.getHyperspaceDistance(
                            game,
                            effHsLvl,
                        );
                    if (hopDist > effHsDist && hopDist <= hsDist) continue;
                }

                const prevSpec = c.specialistId;
                c.specialistId = spec.id;
                const testRes = this.container.combatService.computeStar(
                    game,
                    dstStar,
                    arriving,
                );
                c.specialistId = prevSpec;
                if (!testRes || !myWins(testRes)) continue;

                if (true) {
                    try {
                        await this.container.specialistHireService.hireCarrierSpecialist(
                            game,
                            player,
                            c._id,
                            spec.id,
                            this.container.statisticsService,
                        );
                        const dipped = balance - cost < reserve;
                        if (dipped) {
                            this.activeOptions?.onAlert?.({
                                kind: "emergency_specialist_dip",
                                starIndex: dstIdx,
                                starId: dstStar._id.toString(),
                                starName: dstStar.name,
                                specialistId: spec.id,
                                specialistName: spec.name,
                                ships: c.ships ?? 0,
                                ticksUntil: eta,
                                carrierIds: [c._id.toString()],
                                message: `Emergency specialist dip: hired '${spec.name}' on carrier '${c.name}' (cost ${cost} ${currency === "creditsSpecialists" ? "token(s)" : "cr"}) to swing attack on ${dstStar.name}.`,
                            });
                        }
                        break;
                    } catch (e: any) {
                        if (!(e instanceof ValidationError)) throw e;
                    }
                }
            }
        }
    }

    private _requiredShipsForTarget(
        game: Game,
        player: Player,
        targetStar: Star,
        targetIdx: number,
        ticksToArrival: number,
        attackMargin: number,
        believedIndustry?: (starIndex: number) => number | undefined,
        extraAttackerWeapons = 0,
    ): number {
        const starId = targetStar._id.toString();
        const orbitingEnemy = game.galaxy.carriers.filter(
            (c) =>
                c.orbiting?.toString() === starId &&
                (this.activeOptions?.diplomacy
                    ? !this._isAllied(game, player, c.ownedByPlayerId)
                    : this._isEnemy(game, player, c.ownedByPlayerId)),
        );
        const shipsOnCarriers = orbitingEnemy.reduce(
            (s, c) => s + (c.ships ?? 0),
            0,
        );
        const techs =
            this.container.technologyService.getStarEffectiveTechnologyLevels(
                game,
                targetStar,
                true,
            );
        const ind =
            believedIndustry?.(targetIdx) ??
            targetStar.infrastructure?.industry ??
            0;
        const produced = this.container.shipService.calculateStarShipsByTicks(
            techs.manufacturing,
            ind,
            ticksToArrival,
            game.settings.galaxy.productionTicks,
        );
        const defenderShips = Math.ceil(
            (targetStar.shipsActual ?? targetStar.ships ?? 0) +
                shipsOnCarriers +
                produced,
        );
        const attackerWeapons = Math.max(
            1,
            player.research.weapons.level + extraAttackerWeapons,
        );
        const defBonus =
            game.settings.specialGalaxy.defenderBonus === "enabled";
        const needed =
            this.container.combatService.calculateBasic(
                {
                    ships: defenderShips,
                    weaponsLevel: Math.max(1, techs.weapons),
                },
                {
                    ships: 1,
                    weaponsLevel: attackerWeapons,
                },
                true,
                defBonus,
            ).attacker.shipsNeeded ?? 1;
        return Math.max(1, Math.ceil(needed * attackMargin));
    }

    // Hybrid Coordinated Invasions:
    // - Mode A: 1..3 direct border stars with synchronized Time-on-Target
    //   (`delayTicks <= 8` ticks) to crack fortified targets together.
    // - Mode B: Single-armada multi-hop strike along `ownGraph` (e.g. across
    //   Warp-Gate highways) when a single rear shipyard holds `>= req` ships.
    private async _executeCoordinatedInvasions(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        lockedShips: Float32Array,
        garrisonTarget: Float32Array,
        committedTargets: Map<number, number>,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        enemyRangeByPid: Map<string, number>,
        decision: AdmiralDecision,
        believedIndustry:
            ((starIndex: number) => number | undefined) | undefined,
        starVisibility: Uint8Array | null | undefined,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        upgradeReserve: number,
        maxAttackTicks: number,
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const baseMargin = decision.attackMargin ?? 3.2;
        const directTickCap = Math.min(maxAttackTicks, 26);

        const targets: {
            target: number;
            score: number;
            effMargin: number;
            launchers: { src: number; eta: number }[];
        }[] = [];

        for (let t = 0; t < S; t++) {
            if (!this._isEnemy(game, player, stars[t].ownedByPlayerId))
                continue;
            const desire = decision.attackDesire?.[t] ?? 1.0;
            if (desire <= 0) continue;

            const launchers: { src: number; eta: number }[] = [];
            let minEta = Infinity;
            for (const u of ownStarIndices) {
                if (doomedStars.has(u)) continue;
                if (this.activeBlockedEdges?.has(`${u}->${t}`)) continue;
                if (distMatrix[u * S + t] <= hsDist) {
                    const hopEta = this._hopTicks(
                        game,
                        player,
                        distMatrix,
                        S,
                        u,
                        t,
                    );
                    if (hopEta <= directTickCap) {
                        launchers.push({ src: u, eta: hopEta });
                        if (hopEta < minEta) minEta = hopEta;
                    }
                }
            }
            if (!launchers.length) continue;
            launchers.sort((a, b) => a.eta - b.eta);

            const effMargin =
                starVisibility == null
                    ? baseMargin
                    : starVisibility[t] === VISIBLE
                      ? Math.min(baseMargin, Math.max(1.3, 0.65 * baseMargin))
                      : baseMargin;

            const baseRequired = this._requiredShipsForTarget(
                game,
                player,
                stars[t],
                t,
                minEta,
                1.0,
                believedIndustry,
            );
            if ((committedTargets.get(t) ?? 0) >= baseRequired) continue;

            const value = this._starStrategicValue(stars[t]) * desire;
            const costFactor = Math.max(5, baseRequired) * Math.sqrt(minEta);
            targets.push({
                target: t,
                score: value / costFactor,
                effMargin,
                launchers,
            });
        }

        targets.sort((a, b) => b.score - a.score);

        for (const { target, effMargin, launchers } of targets) {
            const availLaunchers: {
                src: number;
                eta: number;
                surplus: number;
                needsBuy: boolean;
            }[] = [];

            for (const l of launchers) {
                let guard = 0;
                if (decision.garrisonTarget?.[l.src] != null) {
                    guard = decision.garrisonTarget[l.src] * 0.5;
                } else {
                    // Hold a modest local guard if OTHER enemy stars (excluding `target`)
                    // can actually jump onto `l.src` so a neighbor cannot backstab `l.src`.
                    let otherEnemyShips = 0;
                    for (let e = 0; e < S; e++) {
                        const eOwner = stars[e].ownedByPlayerId;
                        if (
                            e === target ||
                            !this._isEnemy(game, player, eOwner)
                        ) {
                            continue;
                        }
                        const eRange =
                            enemyRangeByPid.get(eOwner!.toString()) ?? 0;
                        if (distMatrix[l.src * S + e] <= eRange) {
                            otherEnemyShips += stars[e].ships ?? 0;
                        }
                    }
                    guard = Math.min(22, Math.ceil(otherEnemyShips * 0.15));
                }
                const reserve = Math.max(lockedShips[l.src], guard);
                const surplus = this._availableShipsAt(
                    stars[l.src],
                    idleCarriersAt.get(l.src),
                    reserve,
                );
                if (surplus < 2) continue;
                const hasIdle = (idleCarriersAt.get(l.src)?.length ?? 0) > 0;
                availLaunchers.push({
                    src: l.src,
                    eta: l.eta,
                    surplus,
                    needsBuy: !hasIdle,
                });
            }
            if (!availLaunchers.length) continue;

            // Coordinate at most 3 close border stars (`eta_k - eta_0 <= 8` ticks)
            // so ships aren't locked in long `delayTicks` orbits.
            const maxK = Math.min(availLaunchers.length, 3);
            const firstEta = availLaunchers[0].eta;

            let cumShips = committedTargets.get(target) ?? 0;
            let carriersToBuy = 0;
            let chosenK = -1;
            let syncTick = 0;
            let neededAtSync = 0;
            let useSiegeBreaker = false;

            const canSiege =
                decision.autoCarrierSpecialist !== false &&
                this._canAffordCarrierSpecialist(
                    game,
                    player,
                    SPEC_SIEGE_BREAKER,
                ) &&
                ((player as any)[
                    game.settings.specialGalaxy.specialistsCurrency
                ] ?? 0) >= 6;

            for (let k = 0; k < maxK; k++) {
                const item = availLaunchers[k];
                if (k > 0 && item.eta - firstEta > 8) break;
                if (item.needsBuy) carriersToBuy++;
                const buyCost = carriersToBuy * this._carrierCost(game);
                if (
                    carriersToBuy > 0 &&
                    ((player.credits ?? 0) - upgradeReserve < buyCost ||
                        ownCarrierCount + carriersToBuy >
                            this.container.carrierService.getCarrierLimit(
                                game,
                                player,
                            ))
                ) {
                    break;
                }

                cumShips += item.surplus;
                const tSync = item.eta;
                let req = this._requiredShipsForTarget(
                    game,
                    player,
                    stars[target],
                    target,
                    tSync,
                    effMargin,
                    believedIndustry,
                    0,
                );

                let withSiege = false;
                if (cumShips < req && canSiege && tSync <= 14 && req >= 50) {
                    const siegeReq = this._requiredShipsForTarget(
                        game,
                        player,
                        stars[target],
                        target,
                        tSync,
                        effMargin,
                        believedIndustry,
                        4,
                    );
                    if (cumShips >= siegeReq) {
                        req = siegeReq;
                        withSiege = true;
                    }
                }

                if (cumShips >= req) {
                    chosenK = k;
                    syncTick = tSync;
                    neededAtSync = Math.max(
                        1,
                        req - (committedTargets.get(target) ?? 0),
                    );
                    useSiegeBreaker = withSiege;
                    break;
                }
            }

            if (chosenK < 0) {
                // Mode B fallback: check if a SINGLE rear shipyard `u` can strike
                // through a border jump star along `ownGraph` within `maxAttackTicks`
                // without locking up border garrisons!
                const directSet = new Set(launchers.map((l) => l.src));
                let bestRear = -1;
                let bestBorder = -1;
                let bestRearEta = Infinity;
                let bestRearReq = 0;

                for (const u of ownStarIndices) {
                    if (doomedStars.has(u) || directSet.has(u)) continue;
                    const hasIdle = (idleCarriersAt.get(u)?.length ?? 0) > 0;
                    if (
                        !hasIdle &&
                        !this._canBuildCarrier(
                            game,
                            player,
                            stars[u],
                            ownCarrierCount,
                            upgradeReserve,
                        )
                    ) {
                        continue;
                    }
                    const guard =
                        decision.garrisonTarget?.[u] != null
                            ? decision.garrisonTarget[u] * 0.5
                            : 0;
                    const surplus = this._availableShipsAt(
                        stars[u],
                        idleCarriersAt.get(u),
                        Math.max(lockedShips[u], guard),
                    );
                    if (surplus < 10) continue;

                    for (const b of launchers) {
                        const pTicks = ownGraph.ticks[u * S + b.src];
                        if (!Number.isFinite(pTicks)) continue;
                        const totEta = pTicks + b.eta;
                        if (totEta > maxAttackTicks || totEta >= bestRearEta) {
                            continue;
                        }
                        const req = this._requiredShipsForTarget(
                            game,
                            player,
                            stars[target],
                            target,
                            totEta,
                            effMargin,
                            believedIndustry,
                            0,
                        );
                        const need = Math.max(
                            1,
                            req - (committedTargets.get(target) ?? 0),
                        );
                        if (surplus >= need) {
                            bestRear = u;
                            bestBorder = b.src;
                            bestRearEta = totEta;
                            bestRearReq = need;
                        }
                    }
                }

                if (bestRear >= 0 && bestBorder >= 0) {
                    const prefix = this._buildPathWaypoints(
                        game,
                        bestRear,
                        bestBorder,
                        ownGraph.nextHop,
                        "nothing",
                        false,
                        0,
                    );
                    if (prefix.length > 0 && prefix.length < 28) {
                        prefix.push({
                            source: stars[bestBorder]._id,
                            destination: stars[target]._id,
                            action: "nothing",
                            actionShips: 0,
                            delayTicks: 0,
                        });
                        const res = await this._launchShips(
                            game,
                            player,
                            bestRear,
                            bestRearReq,
                            prefix,
                            idleCarriersAt,
                            ownCarrierCount,
                            upgradeReserve,
                        );
                        if (res.launched) {
                            ownCarrierCount = res.ownCarrierCount;
                            committedTargets.set(
                                target,
                                (committedTargets.get(target) ?? 0) +
                                    res.shipsSent,
                            );
                        }
                    }
                }
                continue;
            }

            let shipsRemaining = neededAtSync;
            let largestCarrier: Carrier | null = null;
            let largestSent = 0;

            for (let k = 0; k <= chosenK && shipsRemaining > 0; k++) {
                const item = availLaunchers[k];
                const send = Math.min(item.surplus, shipsRemaining);
                if (send < 1) continue;
                const delay = Math.max(0, syncTick - item.eta);
                const wps: CarrierWaypointBase<DBObjectId>[] = [
                    {
                        source: stars[item.src]._id,
                        destination: stars[target]._id,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: delay,
                    },
                ];

                const res = await this._launchShips(
                    game,
                    player,
                    item.src,
                    send,
                    wps,
                    idleCarriersAt,
                    ownCarrierCount,
                    upgradeReserve,
                );
                if (res.launched && res.carrier) {
                    ownCarrierCount = res.ownCarrierCount;
                    shipsRemaining -= res.shipsSent;
                    committedTargets.set(
                        target,
                        (committedTargets.get(target) ?? 0) + res.shipsSent,
                    );
                    if (
                        res.shipsSent > largestSent &&
                        res.carrier.specialistId == null
                    ) {
                        largestSent = res.shipsSent;
                        largestCarrier = res.carrier;
                    }
                }
            }

            if (largestCarrier && decision.autoCarrierSpecialist !== false) {
                const specId = useSiegeBreaker
                    ? SPEC_SIEGE_BREAKER
                    : largestSent >= 80 &&
                        this._canAffordCarrierSpecialist(
                            game,
                            player,
                            SPEC_INFILTRATOR,
                        ) &&
                        ((player as any)[
                            game.settings.specialGalaxy.specialistsCurrency
                        ] ?? 0) >= 8
                      ? SPEC_INFILTRATOR
                      : 0;
                if (
                    specId > 0 &&
                    this._canAffordCarrierSpecialist(game, player, specId)
                ) {
                    try {
                        await this.container.specialistHireService.hireCarrierSpecialist(
                            game,
                            player,
                            largestCarrier._id,
                            specId,
                            this.container.statisticsService,
                        );
                    } catch (e: any) {
                        if (!(e instanceof ValidationError)) throw e;
                    }
                }
            }
        }

        return ownCarrierCount;
    }

    private _starShipsPerCycle(game: Game, star: Star): number {
        const ind = star.infrastructure?.industry ?? 0;
        if (ind <= 0) return 0;
        const techs =
            this.container.technologyService.getStarEffectiveTechnologyLevels(
                game,
                star,
                true,
            );
        const prodTicks = game.settings.galaxy.productionTicks;
        return this.container.shipService.calculateStarShipsByTicks(
            techs.manufacturing,
            ind,
            prodTicks,
            prodTicks,
        );
    }

    private _buildLogisticsWaypoints(
        game: Game,
        srcIdx: number,
        dstIdx: number,
        nextHop: Int16Array,
        canSweepStar: (idx: number) => boolean,
    ): {
        waypoints: CarrierWaypointBase<DBObjectId>[];
        sweptIndices: number[];
    } {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const path: number[] = [srcIdx];
        let cur = srcIdx;
        let guard = 0;
        while (cur !== dstIdx && guard++ < 14) {
            const nxt = nextHop[cur * S + dstIdx];
            if (nxt < 0) return { waypoints: [], sweptIndices: [] };
            path.push(nxt);
            cur = nxt;
        }
        if (cur !== dstIdx || path.length < 2) {
            return { waypoints: [], sweptIndices: [] };
        }

        const wps: CarrierWaypointBase<DBObjectId>[] = [];
        const sweptIndices: number[] = [];
        for (let i = 1; i < path.length; i++) {
            const node = path[i];
            const isLast = i === path.length - 1;
            const sweep = !isLast && canSweepStar(node);
            if (sweep) sweptIndices.push(node);
            wps.push({
                source: stars[path[i - 1]]._id,
                destination: stars[node]._id,
                action: isLast ? "dropAll" : sweep ? "collectAll" : "nothing",
                actionShips: 0,
                delayTicks: 0,
            });
        }
        if (wps.length * 2 <= 30) {
            for (let i = path.length - 1; i >= 1; i--) {
                wps.push({
                    source: stars[path[i]]._id,
                    destination: stars[path[i - 1]]._id,
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 0,
                });
            }
        }
        return { waypoints: wps, sweptIndices };
    }

    // Routes surplus ships from interior stars along shortest tick-cost paths
    // (using Warp Gates & Wormholes) toward frontier stars that are below their
    // `garrisonTarget` or staging for enemy fronts, while strictly protecting
    // cycle upgrade cash (`totalEconomy * 5`) and reusing returning ferries.
    private async _executeLogistics(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        lockedShips: Float32Array,
        garrisonTarget: Float32Array,
        hostileBorder: Uint8Array,
        nearbyEnemyShipsAt: Float32Array,
        committedTargets: Map<number, number>,
        activeLogisticsRoutes: Set<string>,
        busyLogisticsStars: Set<number>,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        enemyRangeByPid: Map<string, number>,
        decision: AdmiralDecision,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        logisticsCreditReserve: number,
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;

        // Compute distance from every own star to the nearest enemy star so
        // front-facing perimeter stars ("advance hubs" with no own neighbor
        // closer to an enemy) stage ships forward even before range contact.
        const minEnemyDist = new Float32Array(S).fill(Infinity);
        let anyEnemy = false;
        for (const u of ownStarIndices) {
            for (let t = 0; t < S; t++) {
                if (!this._isEnemy(game, player, stars[t].ownedByPlayerId))
                    continue;
                anyEnemy = true;
                const d = distMatrix[u * S + t];
                if (d < minEnemyDist[u]) minEnemyDist[u] = d;
            }
        }

        const isFrontSink = new Uint8Array(S);
        const demand = new Float32Array(S);
        for (const u of ownStarIndices) {
            if (doomedStars.has(u)) continue;
            const current =
                Math.floor(stars[u].shipsActual ?? stars[u].ships ?? 0) +
                (committedTargets.get(u) ?? 0);
            let want = garrisonTarget[u];

            if (hostileBorder[u] || minEnemyDist[u] <= hsDist) {
                isFrontSink[u] = 1;
                let maxDesire = 0;
                let strikeShips = nearbyEnemyShipsAt[u];
                for (let t = 0; t < S; t++) {
                    const eOwner = stars[t].ownedByPlayerId;
                    if (!this._isEnemy(game, player, eOwner)) continue;
                    const eRange = enemyRangeByPid.get(eOwner!.toString()) ?? 0;
                    const reach = Math.max(hsDist, eRange);
                    if (distMatrix[u * S + t] <= reach) {
                        const d = decision.attackDesire?.[t] ?? 1.0;
                        if (d > maxDesire) maxDesire = d;
                        if (distMatrix[u * S + t] > eRange) {
                            strikeShips += stars[t].ships ?? 0;
                        }
                    }
                }
                want = Math.max(want, 20 + strikeShips * 0.65 + 25 * maxDesire);
            } else if (anyEnemy && Number.isFinite(minEnemyDist[u])) {
                // Advance perimeter hub: no 1-hop safe own neighbor sits closer to the enemy.
                let hasCloserOwnNeighbor = false;
                for (const v of ownStarIndices) {
                    if (v === u || doomedStars.has(v)) continue;
                    if (
                        distMatrix[u * S + v] <= hsDist &&
                        minEnemyDist[v] < minEnemyDist[u] - 0.5
                    ) {
                        hasCloserOwnNeighbor = true;
                        break;
                    }
                }
                if (!hasCloserOwnNeighbor) {
                    isFrontSink[u] = 1;
                    want = Math.max(want, 40);
                }
            }
            demand[u] = Math.max(0, want - current);
        }

        const sinks = ownStarIndices.filter(
            (u) => !doomedStars.has(u) && demand[u] >= 5,
        );
        if (!sinks.length) return ownCarrierCount;

        // Sources: safe interior own stars (not front sinks) with surplus.
        const sources: { src: number; surplus: number; hasIdle: boolean }[] =
            [];
        for (const u of ownStarIndices) {
            if (doomedStars.has(u) || demand[u] > 0 || isFrontSink[u]) continue;
            const reserve = Math.max(lockedShips[u], garrisonTarget[u]);
            const surplus = this._availableShipsAt(
                stars[u],
                idleCarriersAt.get(u),
                reserve,
            );
            const hasIdle = (idleCarriersAt.get(u)?.length ?? 0) > 0;
            if (!hasIdle) {
                if (busyLogisticsStars.has(u)) continue;
                const prodCycle = this._starShipsPerCycle(game, stars[u]);
                const minBuyStockpile = Math.max(8, Math.ceil(prodCycle * 0.7));
                if (surplus < minBuyStockpile) continue;
            } else if (surplus < 4) {
                continue;
            }
            sources.push({ src: u, surplus, hasIdle });
        }
        sources.sort((a, b) => b.surplus - a.surplus);

        const canSweepStar = (m: number) =>
            !isFrontSink[m] &&
            !doomedStars.has(m) &&
            lockedShips[m] <= 0 &&
            garrisonTarget[m] <= 0 &&
            demand[m] <= 0;

        for (const { src, surplus, hasIdle } of sources) {
            if (!hasIdle && busyLogisticsStars.has(src)) continue;

            let bestSink = -1;
            let bestScore = 0;
            for (const dst of sinks) {
                if (dst === src || demand[dst] < 2) continue;
                const eta = ownGraph.ticks[src * S + dst];
                if (!Number.isFinite(eta) || eta <= 0) continue;
                const score = demand[dst] / (eta * eta);
                if (score > bestScore) {
                    bestScore = score;
                    bestSink = dst;
                }
            }
            if (bestSink < 0) continue;

            const hop = ownGraph.nextHop[src * S + bestSink];
            if (hop < 0 || activeLogisticsRoutes.has(`${src}->${hop}`))
                continue;

            const { waypoints, sweptIndices } = this._buildLogisticsWaypoints(
                game,
                src,
                bestSink,
                ownGraph.nextHop,
                canSweepStar,
            );
            if (!waypoints.length) continue;

            const res = await this._launchShips(
                game,
                player,
                src,
                surplus,
                waypoints,
                idleCarriersAt,
                ownCarrierCount,
                logisticsCreditReserve,
            );
            if (res.launched) {
                ownCarrierCount = res.ownCarrierCount;
                demand[bestSink] = Math.max(
                    0,
                    demand[bestSink] - res.shipsSent,
                );
                activeLogisticsRoutes.add(`${src}->${hop}`);
                activeLogisticsRoutes.add(`${hop}->${src}`);
                busyLogisticsStars.add(src);
                for (const m of sweptIndices) {
                    busyLogisticsStars.add(m);
                }
            }
        }

        return ownCarrierCount;
    }
}
