import { createSimContainer } from "../sim/container";
import { fastCalculateBasic } from "../sim/fastCombat";

describe("fastCalculateBasic", () => {
    it("matches CombatService.calculateBasic exactly", () => {
        const combat = createSimContainer().combatService;
        const mismatches: unknown[] = [];
        for (const toStar of [true, false])
            for (const bonus of [true, false])
                for (let wd = 1; wd <= 4; wd++)
                    for (let wa = 1; wa <= 4; wa++)
                        for (const d of [0, 1, 2, 5, 13, 40, 97])
                            for (const a of [0, 1, 3, 8, 21, 60]) {
                                const defender = { ships: d, weaponsLevel: wd };
                                const attacker = { ships: a, weaponsLevel: wa };
                                const want = combat.calculateBasic(defender, attacker, toStar, bonus);
                                const got = fastCalculateBasic(defender, attacker, toStar, bonus);
                                if (JSON.stringify(want) !== JSON.stringify(got)) {
                                    mismatches.push({ d, a, wd, wa, toStar, bonus, want, got });
                                }
                            }
        expect(mismatches).toEqual([]);
    });
});
