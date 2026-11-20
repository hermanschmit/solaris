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

export class TacticalExecutor {
    container: DependencyContainer;

    // Cached static pairwise Euclidean distances per game (non-orbital).
    private distCache = new WeakMap<Game, Float64Array>();

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
        if (otherId.toString() === player._id.toString()) return false;
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

    private _canWarpBetween(
        game: Game,
        player: Player,
        a: Star,
        b: Star,
    ): boolean {
        if (!a.warpGate || !b.warpGate) return false;
        if (!a.ownedByPlayerId || !b.ownedByPlayerId) return false;
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
        if (this.container.starDataService.isStarPairWormHole(stars[i], stars[j])) {
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
        const canTraverse = new Uint8Array(S);
        for (let i = 0; i < S; i++) {
            const owner = stars[i].ownedByPlayerId?.toString();
            if (owner === pid || (allowUnownedIntermediate && !owner)) {
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
    ) {
        if (player.defeated) return;
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

        const prodTicks = game.settings.galaxy.productionTicks;
        const isFirstTick = game.state.tick % prodTicks === 1;
        const isLastTick = game.state.tick % prodTicks === prodTicks - 1;

        // 1. Pool ships on idle orbiting own carriers back onto their stars so
        //    garrison/attack/defense allocation sees one unified ship pool.
        //    Carriers with waypoints (including synchronized `delayTicks > 0`
        //    carriers waiting to launch) are left untouched.
        const idleCarriersAt = new Map<number, Carrier[]>();
        let ownCarrierCount = 0;
        const committedTargets = new Map<number, number>(); // starIdx -> own ships en route
        const outboundLogistics = new Set<string>();

        for (const c of game.galaxy.carriers) {
            if (c.ownedByPlayerId?.toString() !== pid) continue;
            ownCarrierCount++;
            if (c.orbiting && (!c.waypoints || c.waypoints.length === 0)) {
                const idx = starIndexById.get(c.orbiting.toString());
                if (
                    idx !== undefined &&
                    stars[idx].ownedByPlayerId?.toString() === pid
                ) {
                    if ((c.ships ?? 0) > 1) {
                        const add = c.ships! - 1;
                        c.ships = 1;
                        stars[idx].shipsActual =
                            (stars[idx].shipsActual ?? stars[idx].ships ?? 0) +
                            add;
                        stars[idx].ships = Math.floor(stars[idx].shipsActual!);
                    }
                    let list = idleCarriersAt.get(idx);
                    if (!list) {
                        list = [];
                        idleCarriersAt.set(idx, list);
                    }
                    list.push(c);
                }
            } else if (c.waypoints && c.waypoints.length > 0) {
                const destIdx = starIndexById.get(
                    c.waypoints[0].destination.toString(),
                );
                if (destIdx !== undefined) {
                    committedTargets.set(
                        destIdx,
                        (committedTargets.get(destIdx) ?? 0) + (c.ships ?? 0),
                    );
                }
                const srcIdx = starIndexById.get(
                    c.waypoints[0].source.toString(),
                );
                if (srcIdx !== undefined && destIdx !== undefined) {
                    outboundLogistics.add(`${srcIdx}->${destIdx}`);
                }
            }
        }

        const distMatrix = this._pairDistances(game);
        const hsDist = this.container.distanceService.getHyperspaceDistance(
            game,
            player.research.hyperspace.level,
        );
        const ownGraph = this._shortestPathsFromOwn(
            game,
            player,
            distMatrix,
            hsDist,
            false,
        );
        const claimGraph = this._shortestPathsFromOwn(
            game,
            player,
            distMatrix,
            hsDist,
            true,
        );

        // Compute baseline per-star garrison targets (proactive border defense)
        // if not overridden by the policy.
        const maxEnemyRange = Math.max(
            hsDist,
            ...game.galaxy.players
                .filter((p) => !p.defeated && p._id.toString() !== pid)
                .map((p) =>
                    this.container.distanceService.getHyperspaceDistance(
                        game,
                        p.research.hyperspace.level,
                    ),
                ),
        );
        const hostileBorder = new Uint8Array(S);
        const baselineGarrison = new Float32Array(S);
        for (const i of ownStarIndices) {
            let nearbyEnemyShips = 0;
            for (let j = 0; j < S; j++) {
                if (!this._isEnemy(game, player, stars[j].ownedByPlayerId))
                    continue;
                if (distMatrix[i * S + j] <= maxEnemyRange) {
                    hostileBorder[i] = 1;
                    nearbyEnemyShips += stars[j].ships ?? 0;
                }
            }
            if (hostileBorder[i]) {
                baselineGarrison[i] = Math.min(
                    40,
                    Math.ceil(nearbyEnemyShips * 0.2),
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

        // Budget & reserve setup.
        const rawSplit = decision.budgetSplit ?? [0.22, 0.33, 0.3, 0.05, 0.1];
        const splitSum =
            rawSplit.reduce((a, b) => a + Math.max(0, b), 0) || 1.0;
        const normSplit = rawSplit.map((x) => Math.max(0, x) / splitSum);
        const carrierReserve = Math.max(
            this._carrierCost(game),
            Math.floor((player.credits ?? 0) * normSplit[4]),
        );

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
            const neededExtra = this._shipsNeededToHold(
                game,
                player,
                defStar,
                defIdx,
                wave.ticksUntil,
                wave.carriers,
                committedTargets.get(defIdx) ?? 0,
            );

            if (neededExtra <= 0) {
                // Current garrison + arriving ships already hold; lock them!
                lockedShips[defIdx] = Math.max(
                    lockedShips[defIdx],
                    defStar.ships ?? 0,
                );
                continue;
            }

            // Gather own stars that can reach defIdx within wave.ticksUntil.
            const helpers: { src: number; eta: number; avail: number }[] = [];
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
                            0,
                        );
                    if (free > 1 && canCarry) {
                        helpers.push({ src: u, eta, avail: free });
                        totalReachable += free;
                    }
                }
            }

            if (totalReachable < neededExtra * triageThreshold) {
                // Star is doomed! Freeze upgrades here and evacuate garrison
                // right before impact (ticksUntil <= 3) so we keep producing
                // ships until the last safe tick without feeding reinforcements.
                doomedStars.add(defIdx);
                garrisonTarget[defIdx] = 0;
                if (wave.ticksUntil <= 3) {
                    ownCarrierCount = await this._evacuateStar(
                        game,
                        player,
                        defIdx,
                        ownStarIndices,
                        doomedStars,
                        ownGraph,
                        idleCarriersAt,
                        ownCarrierCount,
                    );
                }
                continue;
            }

            // Star is holdable: lock its own garrison and dispatch reinforcements
            // from nearest helpers once slack time <= 1 tick.
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
                    // Hold ships in reserve until launch window arrives.
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
                    const launched = await this._launchShips(
                        game,
                        player,
                        h.src,
                        send,
                        waypoints,
                        idleCarriersAt,
                        ownCarrierCount,
                        0,
                    );
                    if (launched.launched) {
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

        // 3. Targeted infrastructure & Warp Gate spending (skipping doomed stars).
        if (isFirstTick || isLastTick) {
            this._spendCycleCredits(
                game,
                player,
                ownStarIndices,
                doomedStars,
                hostileBorder,
                decision,
                normSplit,
                carrierReserve,
                isFirstTick,
                isLastTick,
            );
        }

        // 4. Claim neutral stars (multi-hop + Explorer outreach).
        const maxAttackTicks = decision.maxAttackTicks ?? prodTicks * 2;
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
            carrierReserve,
            maxAttackTicks,
        );

        // 5. Synchronized Time-on-Target (ToT) multi-star invasions.
        ownCarrierCount = await this._executeCoordinatedInvasions(
            game,
            player,
            ownStarIndices,
            doomedStars,
            lockedShips,
            garrisonTarget,
            committedTargets,
            distMatrix,
            hsDist,
            decision,
            believedIndustry,
            idleCarriersAt,
            ownCarrierCount,
            carrierReserve,
            maxAttackTicks,
        );

        // 6. Proactive border garrisoning & Warp-Gate logistics.
        ownCarrierCount = await this._executeLogistics(
            game,
            player,
            ownStarIndices,
            doomedStars,
            lockedShips,
            garrisonTarget,
            committedTargets,
            outboundLogistics,
            ownGraph,
            distMatrix,
            hsDist,
            decision,
            idleCarriersAt,
            ownCarrierCount,
            carrierReserve,
        );
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
        const grouped = new Map<string, IncomingWave>();
        for (const c of game.galaxy.carriers) {
            if (c.orbiting != null || !c.waypoints?.length) continue;
            if (!this._isEnemy(game, player, c.ownedByPlayerId)) continue;
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
        return [...grouped.values()].sort((a, b) => a.ticksUntil - b.ticksUntil);
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
                c.ownedByPlayerId?.toString() === pid,
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

    private async _evacuateStar(
        game: Game,
        player: Player,
        doomedIdx: number,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
    ): Promise<number> {
        const S = game.galaxy.stars.length;
        const star = game.galaxy.stars[doomedIdx];
        const avail = this._availableShipsAt(
            star,
            idleCarriersAt.get(doomedIdx),
            0,
        );
        if (avail < 2) return ownCarrierCount;

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
            avail,
            waypoints,
            idleCarriersAt,
            ownCarrierCount,
            0,
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
        let remaining = Math.min(budget, player.credits ?? 0);
        while (remaining > 0) {
            let bestIdx = -1;
            let bestScore = 0;
            let bestCost = Infinity;

            for (const idx of candidates) {
                const w = weights(idx);
                if (w <= 0) continue;
                const cost = this._upgradeCost(
                    game,
                    game.galaxy.stars[idx],
                    type,
                );
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

    private _spendCycleCredits(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        hostileBorder: Uint8Array,
        decision: AdmiralDecision,
        normSplit: number[],
        carrierReserve: number,
        isFirstTick: boolean,
        isLastTick: boolean,
    ) {
        const safeOwn = ownStarIndices.filter((i) => !doomedStars.has(i));
        if (!safeOwn.length) return;
        const stars = game.galaxy.stars;

        if (isFirstTick) {
            const spendable = Math.max(
                0,
                (player.credits ?? 0) - carrierReserve,
            );
            const firstShare = normSplit[0] + normSplit[1] + normSplit[3];
            if (spendable <= 0 || firstShare <= 0) return;

            const sciBudget = Math.floor(spendable * normSplit[0]);
            const indBudget = Math.floor(spendable * normSplit[1]);
            const wgBudget = Math.floor(spendable * normSplit[3]);

            // Science: prefer core stars, Research Stations (11) & Binary Stars.
            this._spendPoolOnInfra(
                game,
                player,
                sciBudget,
                "science",
                safeOwn,
                (i) => {
                    const base = decision.sciencePriority?.[i] ?? 1.0;
                    const specBoost = stars[i].specialistId === 11 ? 1.8 : 1.0;
                    const borderPenalty = hostileBorder[i] ? 0.65 : 1.0;
                    return base * specBoost * borderPenalty;
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
                            ? 1.8
                            : 1.0;
                    const borderBoost = hostileBorder[i] ? 1.25 : 1.0;
                    return base * specBoost * borderBoost;
                },
            );

            // Warp Gates: connect high-industry hubs and frontier staging stars.
            let wgRem = Math.min(wgBudget, player.credits ?? 0);
            while (wgRem > 0) {
                let bestIdx = -1;
                let bestScore = 0;
                let bestCost = Infinity;
                for (const i of safeOwn) {
                    const cost = this._warpGateCost(game, stars[i]);
                    if (cost == null || cost <= 0 || cost > wgRem) continue;
                    const w =
                        (decision.warpGatePriority?.[i] ?? 1.0) *
                        (1 + 0.25 * (stars[i].infrastructure?.industry ?? 0));
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
            }
        }

        if (isLastTick) {
            // Right before production, invest remaining credits (keeping a small
            // emergency carrier reserve) into Economy on safe stars.
            const emergencyReserve = Math.floor(this._carrierCost(game));
            const ecoBudget = Math.max(
                0,
                (player.credits ?? 0) - emergencyReserve,
            );
            const ecoFraction =
                normSplit[2] / Math.max(normSplit[2] + normSplit[4], 1e-6);
            this._spendPoolOnInfra(
                game,
                player,
                Math.floor(ecoBudget * Math.max(0.5, ecoFraction)),
                "economy",
                safeOwn,
                (i) => {
                    const base = decision.economyPriority?.[i] ?? 1.0;
                    const borderPenalty = hostileBorder[i] ? 0.25 : 1.0;
                    return base * borderPenalty;
                },
            );
        }
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
        const sds = this.container.starDataService;

        const candidates: { target: number; score: number }[] = [];
        for (let t = 0; t < S; t++) {
            if (stars[t].ownedByPlayerId || sds.isDeadStar(stars[t])) continue;
            if ((committedTargets.get(t) ?? 0) > 0) continue;
            const desire = decision.attackDesire?.[t] ?? 1.0;
            if (desire <= 0) continue;

            let bestEta = Infinity;
            for (const u of ownStarIndices) {
                if (claimGraph.ticks[u * S + t] < bestEta) {
                    bestEta = claimGraph.ticks[u * S + t];
                }
            }
            if (bestEta > maxAttackTicks) continue;
            candidates.push({
                target: t,
                score:
                    (this._starStrategicValue(stars[t]) * desire) /
                    Math.max(1, bestEta),
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
                const reserve = Math.max(lockedShips[u], garrisonTarget[u]);
                const surplus = this._availableShipsAt(
                    stars[u],
                    idleCarriersAt.get(u),
                    reserve,
                );
                const canCarry =
                    (idleCarriersAt.get(u)?.length ?? 0) > 0 ||
                    this._canBuildCarrier(
                        game,
                        player,
                        stars[u],
                        ownCarrierCount,
                        0,
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

            // Send a small scouting detachment (or more if passing through multi-hop free stars).
            const reserve = Math.max(
                lockedShips[bestSrc],
                garrisonTarget[bestSrc],
            );
            const surplus = this._availableShipsAt(
                stars[bestSrc],
                idleCarriersAt.get(bestSrc),
                reserve,
            );
            const send = Math.min(surplus, Math.max(1, wps.length));
            const res = await this._launchShips(
                game,
                player,
                bestSrc,
                send,
                wps,
                idleCarriersAt,
                ownCarrierCount,
                0,
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
        if (decision.autoCarrierSpecialist !== false) {
            ownCarrierCount = await this._dispatchExplorers(
                game,
                player,
                ownStarIndices,
                committedTargets,
                distMatrix,
                hsDist,
                idleCarriersAt,
                ownCarrierCount,
            );
        }

        return ownCarrierCount;
    }

    private async _dispatchExplorers(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        committedTargets: Map<number, number>,
        distMatrix: Float64Array,
        nearDist: number,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
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
                !(committedTargets.get(i) ?? 0)
            ) {
                freeIndices.push(i);
            }
        }
        if (!freeIndices.length) return ownCarrierCount;

        const explorers = game.galaxy.carriers.filter(
            (car) =>
                car.ownedByPlayerId?.toString() === pid &&
                car.specialistId === SPEC_EXPLORER,
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

        // Hire one Explorer if an unclaimed star sits exclusively in (nearDist, farDist].
        let bestPair: { src: number; dst: number; d: number } | null = null;
        for (const t of freeIndices) {
            let minOwnD = Infinity;
            let closestOwn = -1;
            for (const u of ownStarIndices) {
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
        return ((player as any)[currency] ?? 0) >= cost;
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
                this._isEnemy(game, player, c.ownedByPlayerId),
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

    // Synchronized Time-on-Target (ToT) multi-star invasions:
    // Gathers surplus ships from 1..K own stars within direct jump range of an
    // enemy star and sets `delayTicks = T_sync - eta_i` on closer carriers so
    // every attacking fleet lands on the exact same tick and fights together!
    private async _executeCoordinatedInvasions(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        lockedShips: Float32Array,
        garrisonTarget: Float32Array,
        committedTargets: Map<number, number>,
        distMatrix: Float64Array,
        hsDist: number,
        decision: AdmiralDecision,
        believedIndustry: ((starIndex: number) => number | undefined) | undefined,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        carrierReserve: number,
        maxAttackTicks: number,
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;
        const attackMargin = decision.attackMargin ?? 1.35;

        // Score reachable enemy border stars.
        const targets: {
            target: number;
            score: number;
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
                if (distMatrix[u * S + t] <= hsDist) {
                    const eta = this._hopTicks(
                        game,
                        player,
                        distMatrix,
                        S,
                        u,
                        t,
                    );
                    if (eta <= maxAttackTicks) {
                        launchers.push({ src: u, eta });
                        if (eta < minEta) minEta = eta;
                    }
                }
            }
            if (!launchers.length) continue;
            launchers.sort((a, b) => a.eta - b.eta);

            const baseRequired = this._requiredShipsForTarget(
                game,
                player,
                stars[t],
                t,
                minEta,
                1.0,
                believedIndustry,
            );
            // Skip if an existing invasion fleet is already large enough.
            if ((committedTargets.get(t) ?? 0) >= baseRequired) continue;

            const value = this._starStrategicValue(stars[t]) * desire;
            const costFactor = Math.max(5, baseRequired) * Math.sqrt(minEta);
            targets.push({ target: t, score: value / costFactor, launchers });
        }

        targets.sort((a, b) => b.score - a.score);

        for (const { target, launchers } of targets) {
            const availLaunchers: {
                src: number;
                eta: number;
                surplus: number;
                needsBuy: boolean;
            }[] = [];

            for (const l of launchers) {
                const reserve = Math.max(
                    lockedShips[l.src],
                    garrisonTarget[l.src],
                );
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

            // Find minimal prefix k (sorted by eta ascending) whose combined
            // surplus meets the required ship count at synchronized T_sync = eta_k.
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

            for (let k = 0; k < availLaunchers.length; k++) {
                const item = availLaunchers[k];
                if (item.needsBuy) carriersToBuy++;
                const buyCost = carriersToBuy * this._carrierCost(game);
                if (
                    carriersToBuy > 0 &&
                    ((player.credits ?? 0) < buyCost ||
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
                    attackMargin,
                    believedIndustry,
                    0,
                );

                // For heavily fortified targets reachable within 14 ticks (Siege Breaker
                // expires after 15 ticks!), check if +4 weapons lets us crack it cleanly.
                let withSiege = false;
                if (cumShips < req && canSiege && tSync <= 14 && req >= 50) {
                    const siegeReq = this._requiredShipsForTarget(
                        game,
                        player,
                        stars[target],
                        target,
                        tSync,
                        attackMargin,
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

            if (chosenK < 0) continue;

            // Launch synchronized strike from availLaunchers[0..chosenK]!
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
                    0,
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

            // Attach Siege Breaker or Infiltrator on major assaults when token-rich.
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

    // Routes surplus ships from interior stars along shortest tick-cost paths
    // (using Warp Gates & Wormholes) toward frontier stars that are below their
    // `garrisonTarget` or staging for high-`attackDesire` targets.
    private async _executeLogistics(
        game: Game,
        player: Player,
        ownStarIndices: number[],
        doomedStars: Set<number>,
        lockedShips: Float32Array,
        garrisonTarget: Float32Array,
        committedTargets: Map<number, number>,
        outboundLogistics: Set<string>,
        ownGraph: { ticks: Float32Array; nextHop: Int16Array },
        distMatrix: Float64Array,
        hsDist: number,
        decision: AdmiralDecision,
        idleCarriersAt: Map<number, Carrier[]>,
        ownCarrierCount: number,
        carrierReserve: number,
    ): Promise<number> {
        const stars = game.galaxy.stars;
        const S = stars.length;

        // Compute demand at each safe own star: unmet garrisonTarget + offensive
        // staging demand from adjacent high-desire enemy stars.
        const demand = new Float32Array(S);
        for (const u of ownStarIndices) {
            if (doomedStars.has(u)) continue;
            const current =
                Math.floor(stars[u].shipsActual ?? stars[u].ships ?? 0) +
                (committedTargets.get(u) ?? 0);
            let want = garrisonTarget[u];

            for (let t = 0; t < S; t++) {
                if (!this._isEnemy(game, player, stars[t].ownedByPlayerId))
                    continue;
                if (distMatrix[u * S + t] <= hsDist) {
                    const desire = decision.attackDesire?.[t] ?? 1.0;
                    want += Math.min(60, 12 * desire);
                }
            }
            demand[u] = Math.max(0, want - current);
        }

        const sinks = ownStarIndices.filter(
            (u) => !doomedStars.has(u) && demand[u] >= 4,
        );
        if (!sinks.length) return ownCarrierCount;

        // Sources: safe own stars with meaningful surplus above garrisonTarget + lockedShips.
        const minStockpile = Math.max(
            6,
            Math.floor(game.settings.galaxy.productionTicks * 0.4),
        );
        const sources: { src: number; surplus: number }[] = [];
        for (const u of ownStarIndices) {
            if (doomedStars.has(u) || demand[u] > 0) continue;
            const reserve = Math.max(lockedShips[u], garrisonTarget[u]);
            const surplus = this._availableShipsAt(
                stars[u],
                idleCarriersAt.get(u),
                reserve,
            );
            const hasIdle = (idleCarriersAt.get(u)?.length ?? 0) > 0;
            if (surplus >= (hasIdle ? 4 : minStockpile)) {
                sources.push({ src: u, surplus });
            }
        }
        sources.sort((a, b) => b.surplus - a.surplus);

        for (const { src, surplus } of sources) {
            let bestSink = -1;
            let bestScore = 0;
            for (const dst of sinks) {
                if (dst === src) continue;
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
            if (hop < 0 || outboundLogistics.has(`${src}->${hop}`)) continue;

            const wps = this._buildPathWaypoints(
                game,
                src,
                bestSink,
                ownGraph.nextHop,
                "dropAll",
                true,
            );
            if (!wps.length) continue;

            const res = await this._launchShips(
                game,
                player,
                src,
                surplus,
                wps,
                idleCarriersAt,
                ownCarrierCount,
                0,
            );
            if (res.launched) {
                ownCarrierCount = res.ownCarrierCount;
                demand[bestSink] = Math.max(0, demand[bestSink] - res.shipsSent);
                outboundLogistics.add(`${src}->${hop}`);
            }
        }

        return ownCarrierCount;
    }
}
