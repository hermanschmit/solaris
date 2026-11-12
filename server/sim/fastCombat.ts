// The built-in AI sizes attacks and defences with CombatService.calculateBasic:
// one star (or carrier) against one carrier, each with a weapons level. The
// general combat engine answers it by simulating the fight with full combat
// groups, then finds the ships needed by stepping down one ship at a time,
// simulating each, which made it about half of all simulation time. This is
// the same arithmetic on two numbers. sim/spec checks it against
// calculateBasic exactly.

import type { BasicCombatResult } from "@solaris/common";

export interface Side {
    ships: number;
    weaponsLevel: number;
}

// Ships left on each side after the fight. The defender, if a star, gets the
// defender bonus, which it keeps even when it has fewer ships than its
// weapons. Both sides strike at once each round.
function fight(
    defenderShips: number,
    defenderWeapons: number,
    defenderBonus: number,
    attackerShips: number,
    attackerWeapons: number,
): [number, number] {
    let d = defenderShips;
    let a = attackerShips;
    const dTotal = defenderWeapons + defenderBonus;
    while (d > 0 && a > 0) {
        const fromD = d < dTotal ? d + defenderBonus : dTotal;
        const fromA = a < attackerWeapons ? a : attackerWeapons;
        d = Math.max(0, d - fromA);
        a = Math.max(0, a - fromD);
    }
    return [d, a];
}

// The fewest ships (at least `from`) for which ok holds, given that it holds
// for every count above the answer.
function fewest(from: number, ok: (n: number) => boolean): number {
    let hi = Math.max(from, 1);
    while (!ok(hi)) hi *= 2;
    let lo = from - 1; // ok(lo) is false or lo is below the range
    while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (ok(mid)) hi = mid;
        else lo = mid;
    }
    return hi;
}

export function fastCalculateBasic(
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
        fight(d, defender.weaponsLevel, bonus, a, attacker.weaponsLevel);
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
