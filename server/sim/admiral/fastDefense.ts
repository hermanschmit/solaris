// Fast, exact combat sizing for both 2-side basic combat and N-group star
// defense estimation. Replaces linear 1-ship-at-a-time search and per-ship
// carrier damage distribution during bound search with O(1) bulk rounds and
// binary search, matching CombatService results bit-for-bit.

import type {
    BasicCombatResult,
    CombatBaseCarrier,
    CombatBasePlayer,
    CombatBaseStar,
    CombatGroup,
    DetailedCombatResult,
    DetailedCombatResultGroup,
} from "@solaris/common";

export interface Side {
    ships: number;
    weaponsLevel: number;
}

// O(1) exact 1v1 combat simulation. While both sides have at least their full
// weapons strike, damage per round is constant so bulk rounds can be skipped
// in a single division.
export function fastFight(
    defenderShips: number,
    defenderWeapons: number,
    defenderBonus: number,
    attackerShips: number,
    attackerWeapons: number,
): [number, number] {
    let d = defenderShips;
    let a = attackerShips;
    const dTotal = defenderWeapons + defenderBonus;

    if (d >= dTotal && a >= attackerWeapons) {
        const fullRounds = Math.min(
            Math.floor((d - dTotal) / attackerWeapons),
            Math.floor((a - attackerWeapons) / dTotal),
        );
        if (fullRounds > 0) {
            d -= fullRounds * attackerWeapons;
            a -= fullRounds * dTotal;
        }
    }

    while (d > 0 && a > 0) {
        const fromD = d < dTotal ? d + defenderBonus : dTotal;
        const fromA = a < attackerWeapons ? a : attackerWeapons;
        d = Math.max(0, d - fromA);
        a = Math.max(0, a - fromD);
    }
    return [d, a];
}

function fewest(from: number, ok: (n: number) => boolean): number {
    if (ok(from)) return from;
    let step = 1;
    let lo = from;
    let hi = from + step;
    while (!ok(hi)) {
        lo = hi;
        step *= 2;
        hi = from + step;
    }
    while (hi - lo > 1) {
        const mid = lo + Math.floor((hi - lo) / 2);
        if (ok(mid)) hi = mid;
        else lo = mid;
    }
    return hi;
}

export function fastCalculateBasicO1(
    defender: Side,
    attacker: Side,
    isCarrierToStarCombat: boolean,
    includeDefenderBonus: boolean,
): BasicCombatResult {
    if (defender.weaponsLevel <= 0 || attacker.weaponsLevel <= 0) {
        throw new Error(
            `Received invalid weapons levels: ${defender.weaponsLevel}, ${attacker.weaponsLevel}`,
        );
    }
    const bonus = isCarrierToStarCombat && includeDefenderBonus ? 1 : 0;
    const run = (d: number, a: number) =>
        fastFight(d, defender.weaponsLevel, bonus, a, attacker.weaponsLevel);
    const [dAfter, aAfter] = run(defender.ships, attacker.ships);
    const attackerWon = aAfter > dAfter;
    const defenderNeeded = attackerWon
        ? fewest(0, (n) => run(n, attacker.ships)[1] <= 0)
        : undefined;
    const attackerNeeded = attackerWon
        ? undefined
        : fewest(0, (n) => run(defender.ships, n)[1] > 0);
    return {
        defender: {
            shipsBefore: defender.ships,
            shipsAfter: dAfter,
            shipsLost: defender.ships - dAfter,
            shipsNeeded: defenderNeeded,
        },
        attacker: {
            shipsBefore: attacker.ships,
            shipsAfter: aAfter,
            shipsLost: attacker.ships - aAfter,
            shipsNeeded: attackerNeeded,
        },
    };
}

// Simulates N-group combat tracking only group ship totals (no Map allocations
// or per-ship carrier distribution), which is sufficient to determine every
// group's shipsAfter.
function simulateGroupShips<
    ID,
    P extends CombatBasePlayer<ID>,
    S extends CombatBaseStar<ID>,
    C extends CombatBaseCarrier<ID>,
>(
    groups: CombatGroup<ID, P, S, C>[],
    targetIdx: number,
    overrideShips: number,
): number[] {
    const n = groups.length;
    if (n === 2) {
        const g0 = groups[0];
        const g1 = groups[1];
        const s0 = targetIdx === 0 ? overrideShips : g0.originalShips;
        const s1 = targetIdx === 1 ? overrideShips : g1.originalShips;
        const w0 = g0.attackAgainst.get(1)!;
        const w1 = g1.attackAgainst.get(0)!;
        const b0 = g0.isDefender ? w0.defenderBonus || 0 : 0;
        const b1 = g1.isDefender ? w1.defenderBonus || 0 : 0;
        // In carrier-to-star combat, group 0 is always the star defender.
        if (b1 === 0) {
            const [rem0, rem1] = fastFight(
                s0,
                w0.total - b0,
                b0,
                s1,
                w1.total,
            );
            return [rem0, rem1];
        }
    }

    const ships = groups.map((g, i) =>
        i === targetIdx ? overrideShips : g.originalShips,
    );
    const next = new Array<number>(n);

    while (true) {
        let alive = 0;
        for (let i = 0; i < n; i++) {
            if (ships[i] > 0) alive++;
        }
        if (alive <= 1) break;

        for (let i = 0; i < n; i++) {
            let incoming = 0;
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const w = groups[j].attackAgainst.get(i)!;
                let dmg: number;
                if (ships[j] < w.total) {
                    dmg = groups[j].isDefender
                        ? ships[j] + (w.defenderBonus || 0)
                        : ships[j];
                } else {
                    dmg = w.total;
                }
                incoming += dmg;
            }
            next[i] = Math.max(0, ships[i] - incoming);
        }
        for (let i = 0; i < n; i++) ships[i] = next[i];
    }
    return ships;
}

export function fastEstimateNeeded<
    ID,
    P extends CombatBasePlayer<ID>,
    S extends CombatBaseStar<ID>,
    C extends CombatBaseCarrier<ID>,
>(
    combatResult: DetailedCombatResult<ID, P, S, C>,
    estimateForGroup: DetailedCombatResultGroup<ID, P, S, C>,
    terminationCondition:
        | "greaterThanZeroShips"
        | "eliminateOtherGroups" = "greaterThanZeroShips",
): number {
    const groups = combatResult.combatGroups;
    const targetIdx = groups.findIndex((g) => g.id === estimateForGroup.id);
    if (targetIdx < 0) return estimateForGroup.shipsAfter;

    const targetGroup = groups[targetIdx];
    const baseObjShips = targetGroup.star
        ? (targetGroup.star.ships as number)
        : (targetGroup.carriers[0]?.ships as number) ?? 0;
    // Minimum shipsNeeded that keeps modified star/carrier ships >= 0.
    const minValid = Math.max(0, targetGroup.originalShips - baseObjShips);

    const satisfies = (candidateShips: number): boolean => {
        const after = simulateGroupShips(groups, targetIdx, candidateShips);
        if (terminationCondition === "greaterThanZeroShips") {
            return after[targetIdx] > 0;
        }
        for (let i = 0; i < after.length; i++) {
            if (i !== targetIdx && after[i] > 0) return false;
        }
        return true;
    };

    const start = Math.max(minValid, estimateForGroup.shipsAfter);
    return fewest(start, satisfies);
}

// Patch a CombatService instance in-place with fast, memoized O(1)/O(log N)
// combat estimators. Results are identical to CombatService.
export function installFastCombat(combatService: any) {
    const basicCache = new Map<string, BasicCombatResult>();
    combatService.calculateBasic = (
        defender: Side,
        attacker: Side,
        toStar: boolean,
        bonus: boolean,
    ) => {
        const key = `${defender.ships},${defender.weaponsLevel},${attacker.ships},${attacker.weaponsLevel},${toStar},${bonus}`;
        let answer = basicCache.get(key);
        if (answer === undefined) {
            if (basicCache.size > 100_000) basicCache.clear();
            answer = fastCalculateBasicO1(defender, attacker, toStar, bonus);
            basicCache.set(key, answer);
        }
        return answer;
    };

    combatService.estimateNeeded = (
        combatResult: any,
        estimateForGroup: any,
        terminationCondition:
            | "greaterThanZeroShips"
            | "eliminateOtherGroups" = "greaterThanZeroShips",
    ) =>
        fastEstimateNeeded(
            combatResult,
            estimateForGroup,
            terminationCondition,
        );
}
