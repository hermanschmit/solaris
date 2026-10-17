// Turns a player's view of a game (the JSON returned by
// GET /api/game/:gameId/galaxy) into a complete game the engine can run.
//
// The view hides most enemy detail outside scanning range. Hidden values are
// filled in from what the view does expose (each player's totals in `stats`
// and their research levels). Each call with a different seed gives one
// plausible version of the hidden state, so an advisor can average over
// several of them.
import { Game } from "../services/types/Game";
import { SimEngine } from "./engine";
import { Rng, seededRng } from "./random";

const id = (v: any) => (v == null ? null : v.toString());

// Split `total` into `n` non-negative integers that sum to it, randomly.
const splitInteger = (total: number, n: number, rng: Rng): number[] => {
    if (n === 0) return [];
    total = Math.max(0, Math.round(total));
    const weights = Array.from({ length: n }, () => 0.5 + rng());
    const sum = weights.reduce((a, b) => a + b, 0);
    const parts = weights.map((w) => Math.floor((w / sum) * total));
    let remainder = total - parts.reduce((a, b) => a + b, 0);
    for (let i = 0; remainder > 0; i = (i + 1) % n, remainder--) parts[i]++;
    return parts;
};

export interface DeterminizeOptions {
    seed?: number;
    // Players other than the viewer are handed to the built-in AI unless
    // this is false, in which case they keep a placeholder userId and stay
    // idle unless something else gives them orders.
    opponentsUseBuiltInAI?: boolean;
}

export const determinize = (
    engine: SimEngine,
    snapshot: any,
    options: DeterminizeOptions = {},
): Game => {
    const rng = seededRng(options.seed ?? 1);
    const json = JSON.parse(JSON.stringify(snapshot));
    const galaxy = json.galaxy;

    const viewer = galaxy.players.find((p: any) => p.userId != null);
    const viewerId = id(viewer?._id);

    // Natural resources of hidden stars: assume the average of the stars we
    // can see.
    const seen = galaxy.stars.filter((s: any) => s.naturalResources);
    const avgNR = (key: string) =>
        Math.round(
            seen.reduce((a: number, s: any) => a + s.naturalResources[key], 0) /
                Math.max(1, seen.length),
        );
    const defaultNR = {
        economy: avgNR("economy"),
        industry: avgNR("industry"),
        science: avgNR("science"),
    };

    for (const star of galaxy.stars) {
        star.naturalResources ??= { ...defaultNR };
        star.homeStar ??= galaxy.players.some(
            (p: any) => id(p.homeStarId) === id(star._id),
        );
        star.canBeLooted ??= true;
        star.specialistId ??= null;
        star.specialistExpireTick ??= null;
    }

    for (const player of galaxy.players) {
        const pid = id(player._id);
        if (pid === viewerId) continue;

        const stats = player.stats ?? {};
        const owned = galaxy.stars.filter(
            (s: any) => id(s.ownedByPlayerId) === pid,
        );
        const hidden = owned.filter((s: any) => s.infrastructure == null);
        const visible = owned.filter((s: any) => s.infrastructure != null);
        const visibleCarriers = galaxy.carriers.filter(
            (c: any) => id(c.ownedByPlayerId) === pid,
        );

        // Infrastructure: what the totals say minus what we can see.
        const remaining = (key: string, statKey: string) =>
            (stats[statKey] ?? 0) -
            visible.reduce((a: number, s: any) => a + s.infrastructure[key], 0);
        const eco = splitInteger(
            remaining("economy", "totalEconomy"),
            hidden.length,
            rng,
        );
        const ind = splitInteger(
            remaining("industry", "totalIndustry"),
            hidden.length,
            rng,
        );
        const sci = splitInteger(
            remaining("science", "totalScience"),
            hidden.length,
            rng,
        );

        // Ships: total minus everything visible, garrisoned on hidden stars.
        // Hidden carriers are not invented; their ships sit at stars instead.
        const visibleShips =
            visible.reduce((a: number, s: any) => a + (s.ships ?? 0), 0) +
            visibleCarriers.reduce(
                (a: number, c: any) => a + (c.ships ?? 0),
                0,
            );
        const ships = splitInteger(
            (stats.totalShips ?? 0) - visibleShips,
            hidden.length,
            rng,
        );

        hidden.forEach((s: any, i: number) => {
            s.infrastructure = {
                economy: eco[i],
                industry: ind[i],
                science: sci[i],
            };
            s.ships = ships[i];
            s.shipsActual = ships[i];
        });

        // Research levels are public; progress and credits are not.
        for (const tech of Object.keys(player.research ?? {})) {
            player.research[tech].progress ??= 0;
        }
        player.researchingNow ??= "weapons";
        player.researchingNext ??= "weapons";
        player.credits ??= 0;
        player.creditsSpecialists ??= 0;
        player.renownToGive ??= 0;
        player.userId =
            options.opponentsUseBuiltInAI === false ? viewer?.userId : null;
    }

    for (const star of galaxy.stars) {
        star.shipsActual ??= star.ships ?? 0;
        star.ships ??= Math.floor(star.shipsActual);
        star.infrastructure ??= { economy: 0, industry: 0, science: 0 };
    }

    return engine.fromJSON(json);
};
