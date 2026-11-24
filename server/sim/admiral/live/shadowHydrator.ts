import { DiplomaticStatus } from "@solaris/common";
import { SimEngine } from "../../engine";
import { SeatMemory, seatFrame } from "../../fog";
import { HIDDEN, SeatView, VISIBLE, VISIBLE_NO_SHIPS } from "../../beliefData";
import { AdmiralObservation, AdmiralOwnFacts } from "../admiralEnv";
import { Game } from "../../../services/types/Game";
import { Star, InfrastructureType } from "../../../services/types/Star";
import { Carrier } from "../../../services/types/Carrier";
import { DiplomaticPolicy } from "./diplomaticPolicy";

export interface HydratedShadowContext {
    shadowGame: Game;
    myPlayer: any;
    mySeatIndex: number;
    technologies: string[];
    view: SeatView;
    observation: AdmiralObservation;
    staticHeader: any;
    rawPublicStats: number[][];
}

const strId = (v: any): string | null => (v == null ? null : v.toString());

/**
 * Hydrates an in-memory `Game`, `SeatView`, `SeatMemory`, `AdmiralObservation`,
 * and `BeliefRecorder`-compatible static header from live `/galaxy` + `/diplomacy`
 * JSON responses.
 *
 * Handles live fog/nebula/stealth edge cases verified against `gameGalaxy.ts`:
 * - `s.isInScanningRange && s.ships === null` -> `VISIBLE_NO_SHIPS` (not `0`!)
 * - Nebula stars in scanning range have `naturalResources` stripped by `gameGalaxy.ts`;
 *   we populate placeholder non-zero `naturalResources` so `isDeadStar` never misclassifies
 *   a nebula as a dead star!
 * - Public scoreboard totals (`player.stats`) from `/galaxy` override `frame.stats`
 *   so `BeliefTracker`'s `PublicTotal - VisibleKnown` conservation is 100% exact!
 */
export function hydrateShadowFromGalaxy(params: {
    engine: SimEngine;
    galaxyPayload: any;
    diplomaticStatuses?: DiplomaticStatus<string>[] | null;
    seatMemory: SeatMemory;
    diplomacy?: DiplomaticPolicy | null;
    monthlyBannedStarSpecs?: number[];
    monthlyBannedCarrierSpecs?: number[];
}): HydratedShadowContext {
    const { engine, seatMemory } = params;
    const raw = JSON.parse(JSON.stringify(params.galaxyPayload));
    const rawGalaxy = raw.galaxy;

    // Identify our player: in `/galaxy`, only the logged-in user's player has `userId != null`
    // (or `hasPerspective === true` / `credits !== undefined`).
    let mySeatIndex = rawGalaxy.players.findIndex(
        (p: any) => p.hasPerspective === true || p.userId != null,
    );
    if (mySeatIndex < 0 && params.diplomacy) {
        mySeatIndex = rawGalaxy.players.findIndex(
            (p: any) => p._id?.toString() === params.diplomacy!.myPlayerId,
        );
    }
    if (mySeatIndex < 0) {
        throw new Error(
            "Unable to locate authenticated player seat in /galaxy response.",
        );
    }
    const rawMe = rawGalaxy.players[mySeatIndex];
    const myId = rawMe._id.toString();

    // Capture authoritative public scoreboard stats `[stars, ships, carriers, eco, ind, sci]`
    // BEFORE `fromJSON` fills defaults on hidden stars.
    const rawPublicStats: number[][] = rawGalaxy.players.map((p: any) => {
        if (p.stats) {
            return [
                Number(p.stats.totalStars ?? 0),
                Number(p.stats.totalShips ?? 0),
                Number(p.stats.totalCarriers ?? 0),
                Number(p.stats.totalEconomy ?? 0),
                Number(p.stats.totalIndustry ?? 0),
                Number(p.stats.totalScience ?? 0),
            ];
        }
        return [0, 0, 0, 0, 0, 0];
    });

    // Track per-star & per-carrier visibility from raw `/galaxy` fields BEFORE schema defaults!
    const rawStars: any[] = rawGalaxy.stars;
    const S = rawStars.length;
    const starVis = new Uint8Array(S).fill(HIDDEN);
    const carrierShipsVisibleById = new Set<string>();

    // Average natural resources of visible non-dead stars for fogged stars.
    const seenWithNR = rawStars.filter(
        (s) =>
            s.naturalResources &&
            (s.naturalResources.economy > 0 ||
                s.naturalResources.industry > 0 ||
                s.naturalResources.science > 0),
    );
    const avgNR = (k: "economy" | "industry" | "science") =>
        seenWithNR.length > 0
            ? Math.max(
                  1,
                  Math.round(
                      seenWithNR.reduce(
                          (acc, s) => acc + Number(s.naturalResources[k] ?? 20),
                          0,
                      ) / seenWithNR.length,
                  ),
              )
            : 25;
    const defaultNR = {
        economy: avgNR("economy"),
        industry: avgNR("industry"),
        science: avgNR("science"),
    };

    rawStars.forEach((s, i) => {
        const ownedByMe = strId(s.ownedByPlayerId) === myId;
        const inScan = ownedByMe || Boolean(s.isInScanningRange);
        if (inScan) {
            starVis[i] =
                ownedByMe || s.ships != null ? VISIBLE : VISIBLE_NO_SHIPS;
        } else {
            starVis[i] = HIDDEN;
        }

        // Preserve genuine dead stars (`naturalResources` explicitly `{0, 0, 0}`),
        // while filling missing `naturalResources` on fogged or scanned nebula stars!
        if (!s.naturalResources) {
            s.naturalResources = s.isNebula
                ? { economy: 25, industry: 25, science: 25 }
                : { ...defaultNR };
        }
        s.homeStar ??= rawGalaxy.players.some(
            (p: any) => strId(p.homeStarId) === strId(s._id),
        );
        s.canBeLooted ??= true;
        s.specialistId ??= null;
        s.specialistExpireTick ??= null;

        if (starVis[i] === VISIBLE) {
            const sh = Number(s.shipsActual ?? s.ships ?? 0);
            s.shipsActual = sh;
            s.ships = Math.floor(sh);
        } else {
            // Remembered or placeholder ships (overwritten by BeliefTracker during execution).
            const memSh = Math.max(0, Math.round(seatMemory.ships[i] ?? 0));
            s.shipsActual = memSh;
            s.ships = memSh;
        }

        if (!s.infrastructure) {
            if (seatMemory.seenTick[i] >= 0 || starVis[i] !== HIDDEN) {
                s.infrastructure = {
                    economy: Math.round(seatMemory.infrastructure[i * 3] ?? 0),
                    industry: Math.round(
                        seatMemory.infrastructure[i * 3 + 1] ?? 0,
                    ),
                    science: Math.round(
                        seatMemory.infrastructure[i * 3 + 2] ?? 0,
                    ),
                };
            } else {
                s.infrastructure = { economy: 0, industry: 0, science: 0 };
            }
        }
    });

    for (const c of rawGalaxy.carriers ?? []) {
        const cid = c._id.toString();
        const ownedByMe = strId(c.ownedByPlayerId) === myId;
        if (ownedByMe || c.ships != null) {
            carrierShipsVisibleById.add(cid);
            c.ships = Math.max(1, Number(c.ships ?? 1));
        } else {
            // Scrambler / Nebula carrier with `ships: null`: keep non-zero placeholder
            // on the model object while omitting from `view.carrierShips` so `seatFrame`
            // reports `-1` (hidden ships) accurately!
            c.ships = 10;
        }
        c.specialistId ??= null;
        c.specialistExpireTick ??= null;
        c.isGift ??= false;
        c.waypointsLooped ??= false;
    }

    // Sync authoritative `/diplomacy` (`statusTo` & `statusFrom`) onto player diplomacy arrays.
    const statusByOpponent = new Map<string, DiplomaticStatus<string>>();
    for (const st of params.diplomaticStatuses ?? []) {
        statusByOpponent.set(st.playerIdTo.toString(), st);
    }
    rawMe.diplomacy ??= [];

    for (const p of rawGalaxy.players) {
        const pid = p._id.toString();
        for (const tech of Object.keys(p.research ?? {})) {
            p.research[tech].progress ??= 0;
        }
        p.researchingNow ??= "weapons";
        p.researchQueue ??= p.researchingNext
            ? [p.researchingNext]
            : ["weapons"];
        p.credits ??= 0;
        p.creditsSpecialists ??= 0;
        p.renownToGive ??= 0;
        p.diplomacy ??= [];

        if (pid !== myId) {
            const st = statusByOpponent.get(pid);
            if (st) {
                const meEntry = rawMe.diplomacy.find(
                    (d: any) => strId(d.playerId) === pid,
                );
                if (meEntry) meEntry.status = st.statusTo;
                else
                    rawMe.diplomacy.push({
                        playerId: pid,
                        status: st.statusTo,
                    });

                const oppEntry = p.diplomacy.find(
                    (d: any) => strId(d.playerId) === myId,
                );
                if (oppEntry) oppEntry.status = st.statusFrom;
                else
                    p.diplomacy.push({ playerId: myId, status: st.statusFrom });
            }
        }
    }

    const shadowGame = engine.fromJSON(raw);
    // Preserve `isInScanningRange` and `ignoreBulkUpgrade` from raw `/galaxy` payload
    // in case Mongoose schema stripped non-schema view fields!
    shadowGame.galaxy.players.forEach((p, idx) => {
        if (rawGalaxy.players[idx]?.isInScanningRange !== undefined) {
            (p as any).isInScanningRange =
                rawGalaxy.players[idx].isInScanningRange;
        }
    });
    shadowGame.galaxy.stars.forEach((s, idx) => {
        if (rawStars[idx]?.ignoreBulkUpgrade) {
            s.ignoreBulkUpgrade = rawStars[idx].ignoreBulkUpgrade;
        }
    });

    const myPlayer = shadowGame.galaxy.players[mySeatIndex];

    const carriersInView = new Set<Carrier>(shadowGame.galaxy.carriers);
    const carrierShipsInView = new Set<Carrier>(
        shadowGame.galaxy.carriers.filter((c) =>
            carrierShipsVisibleById.has(c._id.toString()),
        ),
    );
    const view: SeatView = {
        stars: starVis,
        carriers: carriersInView,
        carrierShips: carrierShipsInView,
    };

    seatMemory.update(engine, shadowGame, myPlayer, view);

    const startLevels =
        shadowGame.settings.technology.startingTechnologyLevel ?? {};
    const technologies = Object.keys(startLevels).filter((k) =>
        /^[^_$]/.test(k),
    );

    const frame = seatFrame(shadowGame, seatMemory, technologies);
    // Override `frame.stats` with authoritative public `player.stats` totals from `/galaxy`
    // whenever `stats` was provided on the player payload.
    rawGalaxy.players.forEach((p: any, idx: number) => {
        if (p.stats) {
            frame.stats[idx] = rawPublicStats[idx].slice();
        }
    });
    // Also sum our own known garrison + carrier ships if `rawMe.stats` wasn't populated.
    if (!rawMe.stats) {
        let myStars = 0;
        let myShips = 0;
        let myEco = 0;
        let myInd = 0;
        let mySci = 0;
        for (const s of shadowGame.galaxy.stars) {
            if (s.ownedByPlayerId?.toString() !== myId) continue;
            myStars++;
            myShips += Math.floor(s.shipsActual ?? s.ships ?? 0);
            myEco += s.infrastructure?.economy ?? 0;
            myInd += s.infrastructure?.industry ?? 0;
            mySci += s.infrastructure?.science ?? 0;
        }
        let myCarriers = 0;
        for (const c of shadowGame.galaxy.carriers) {
            if (c.ownedByPlayerId?.toString() !== myId) continue;
            myCarriers++;
            myShips += c.ships ?? 0;
        }
        frame.stats[mySeatIndex] = [
            myStars,
            myShips,
            myCarriers,
            myEco,
            myInd,
            mySci,
        ];
    }

    const ownFacts = computeLiveOwnFacts({
        engine,
        game: shadowGame,
        player: myPlayer,
        technologies,
        view,
        diplomacy: params.diplomacy ?? null,
        monthlyBannedStarSpecs: params.monthlyBannedStarSpecs,
        monthlyBannedCarrierSpecs: params.monthlyBannedCarrierSpecs,
    });

    const prodTicks = shadowGame.settings.galaxy.productionTicks || 20;
    const tick = shadowGame.state.tick;
    const observation: AdmiralObservation = {
        seat: mySeatIndex,
        tick,
        cycle: Math.floor(tick / prodTicks),
        ticksToProduction: prodTicks - (tick % prodTicks),
        ticksToEnd: shadowGame.state.ticksToEnd ?? null,
        defeated: Boolean(myPlayer.defeated),
        frame,
        ownFacts,
    };

    const staticHeader = buildStaticHeader(engine, shadowGame, technologies);

    return {
        shadowGame,
        myPlayer,
        mySeatIndex,
        technologies,
        view,
        observation,
        staticHeader,
        rawPublicStats,
    };
}

export function buildStaticHeader(
    engine: SimEngine,
    game: Game,
    technologies: string[],
) {
    const distance = engine.container.distanceService;
    const levels = Array.from({ length: 30 }, (_, i) => i + 1);
    return {
        version: 1,
        productionTicks: game.settings.galaxy.productionTicks,
        carrierSpeed: game.settings.specialGalaxy.carrierSpeed,
        hyperspaceRange: levels.map((l) =>
            distance.getHyperspaceDistance(game, l),
        ),
        scanningRange: levels.map((l) => distance.getScanningDistance(game, l)),
        players: game.galaxy.players.map((p) => ({ alias: p.alias })),
        technologies,
        researchable: technologies.filter((t) =>
            engine.container.technologyService.isTechnologyResearchable(
                game,
                t as any,
            ),
        ),
        stars: game.galaxy.stars.map((s) => ({
            x: s.location.x,
            y: s.location.y,
            homeStar: game.galaxy.players.findIndex(
                (p) => p.homeStarId?.toString() === s._id.toString(),
            ),
            naturalResources: s.naturalResources,
            isNebula: Boolean(s.isNebula),
            isPulsar: Boolean(s.isPulsar),
            isBlackHole: Boolean(s.isBlackHole),
            isAsteroidField: Boolean(s.isAsteroidField),
            isBinaryStar: Boolean(s.isBinaryStar),
            hasWormHole: Boolean(s.wormHoleToStarId),
        })),
    };
}

export function computeLiveOwnFacts(params: {
    engine: SimEngine;
    game: Game;
    player: any;
    technologies: string[];
    view: SeatView;
    diplomacy: DiplomaticPolicy | null;
    monthlyBannedStarSpecs?: number[];
    monthlyBannedCarrierSpecs?: number[];
}): AdmiralOwnFacts {
    const { engine, game, player, technologies, view, diplomacy } = params;
    const c = engine.container;
    const pid = player._id.toString();
    const stars = game.galaxy.stars;
    const S = stars.length;
    const starIndexById = new Map<string, number>();
    for (let i = 0; i < S; i++) {
        starIndexById.set(stars[i]._id.toString(), i);
    }

    const upgradeCost = (star: Star, type: InfrastructureType): number => {
        const key = game.settings.player.developmentCost[type];
        if (key === "none" || c.starDataService.isDeadStar(star)) return -1;
        if ((star.ignoreBulkUpgrade as any)?.[type]) return -1;
        const ic = c.infrastructureCostService;
        const fn =
            type === "economy"
                ? ic.calculateEconomyCost.bind(ic)
                : type === "industry"
                  ? ic.calculateIndustryCost.bind(ic)
                  : ic.calculateScienceCost.bind(ic);
        return (
            c.starUpgradeService._calculateUpgradeInfrastructureCost(
                game,
                star,
                key,
                type,
                fn,
            ) ?? -1
        );
    };

    const warpCost = (star: Star): number => {
        const key = game.settings.specialGalaxy.warpgateCost;
        if (
            key === "none" ||
            star.warpGate ||
            c.starDataService.isDeadStar(star)
        ) {
            return -1;
        }
        const mult = game.constants.star.infrastructureExpenseMultipliers[key];
        const techs = c.technologyService.getStarEffectiveTechnologyLevels(
            game,
            star,
        );
        const tr = c.starService.calculateTerraformedResources(
            star,
            techs.terraforming,
        );
        const avg =
            c.infrastructureCostService.calculateAverageTerraformedResources(
                tr,
            );
        return (
            c.infrastructureCostService.calculateWarpGateCost(
                game,
                mult,
                avg,
            ) ?? -1
        );
    };

    const ecoCosts = new Array<number>(S).fill(-1);
    const indCosts = new Array<number>(S).fill(-1);
    const sciCosts = new Array<number>(S).fill(-1);
    const wgCosts = new Array<number>(S).fill(-1);
    for (let i = 0; i < S; i++) {
        if (stars[i].ownedByPlayerId?.toString() !== pid) continue;
        ecoCosts[i] = upgradeCost(stars[i], "economy");
        indCosts[i] = upgradeCost(stars[i], "industry");
        sciCosts[i] = upgradeCost(stars[i], "science");
        wgCosts[i] = warpCost(stars[i]);
    }

    const rps = c.researchService.researchProgressService;
    const researchProgress = technologies.map((t) =>
        Number((player.research as any)?.[t]?.progress ?? 0),
    );
    const researchRequired = technologies.map((t) => {
        const level = Number((player.research as any)?.[t]?.level ?? 1);
        if (!c.technologyService.isTechnologyResearchable(game, t as any)) {
            return 0;
        }
        return Number(
            rps.getRequiredResearchProgress(game, t as any, level) ?? 0,
        );
    });

    const specCurrency = game.settings.specialGalaxy.specialistsCurrency;
    const specBudget =
        game.settings.specialGalaxy.specialistCost === "none"
            ? 0
            : Number((player as any)[specCurrency] ?? 0);

    const extraStarBans = new Set(params.monthlyBannedStarSpecs ?? []);
    const extraCarrierBans = new Set(params.monthlyBannedCarrierSpecs ?? []);

    const affordableStarSpecialists =
        game.settings.specialGalaxy.specialistCost === "none"
            ? []
            : c.specialistService
                  .listStar(game)
                  .filter(
                      (s) =>
                          !c.specialistBanService.isStarSpecialistBanned(
                              game,
                              s.id,
                          ) &&
                          !extraStarBans.has(s.id) &&
                          c.specialistService.getSpecialistActualCost(game, s)[
                              specCurrency
                          ] <= specBudget,
                  )
                  .map((s) => s.id)
                  .sort((a, b) => a - b);

    const affordableCarrierSpecialists =
        game.settings.specialGalaxy.specialistCost === "none"
            ? []
            : c.specialistService
                  .listCarrier(game)
                  .filter(
                      (s) =>
                          !c.specialistBanService.isCarrierSpecialistBanned(
                              game,
                              s.id,
                          ) &&
                          !extraCarrierBans.has(s.id) &&
                          c.specialistService.getSpecialistActualCost(game, s)[
                              specCurrency
                          ] <= specBudget,
                  )
                  .map((s) => s.id)
                  .sort((a, b) => a - b);

    const ownCarriers = game.galaxy.carriers
        .filter((car) => car.ownedByPlayerId?.toString() === pid)
        .map((car) => ({
            id: car._id.toString(),
            ships: car.ships ?? 0,
            orbiting: car.orbiting
                ? (starIndexById.get(car.orbiting.toString()) ?? -1)
                : -1,
            destination: car.waypoints?.length
                ? (starIndexById.get(car.waypoints[0].destination.toString()) ??
                  -1)
                : -1,
            delayTicks: car.waypoints?.length
                ? (car.waypoints[0].delayTicks ?? 0)
                : 0,
            specialist: car.specialistId ?? 0,
        }));

    // Incoming attacks exclude own carriers, `isGift` carriers, and `ALLIED` carriers!
    const attackMap = new Map<
        string,
        { starIndex: number; ticksUntil: number; ships: number }
    >();
    for (const car of game.galaxy.carriers) {
        if (car.orbiting != null || !car.waypoints?.length) continue;
        if (car.isGift) continue;
        const ownerStr = car.ownedByPlayerId?.toString();
        if (!ownerStr || ownerStr === pid) continue;
        if (diplomacy?.isAllied(ownerStr)) continue;
        if (!view.carriers.has(car)) continue;

        const wp = car.waypoints[0];
        const dstIdx = starIndexById.get(wp.destination.toString());
        if (dstIdx === undefined) continue;
        if (stars[dstIdx].ownedByPlayerId?.toString() !== pid) continue;

        const eta =
            c.waypointService.calculateWaypointTicksEta(game, car, wp) || 1;
        const ships = view.carrierShips.has(car) ? (car.ships ?? 1) : 1;
        const key = `${dstIdx}:${eta}`;
        const prev = attackMap.get(key);
        if (prev) prev.ships += ships;
        else attackMap.set(key, { starIndex: dstIdx, ticksUntil: eta, ships });
    }

    const carrierMult =
        game.constants.star.infrastructureExpenseMultipliers[
            game.settings.specialGalaxy.carrierCost
        ];
    const carrierCost = c.infrastructureCostService.calculateCarrierCost(
        game,
        carrierMult,
    );

    return {
        credits: Number(player.credits ?? 0),
        specialistBudget: specBudget,
        carrierCost,
        researchingNow: technologies.indexOf(player.researchingNow),
        researchingNext: technologies.indexOf(
            (player.researchQueue?.[0] ?? "") as string,
        ),
        researchProgress,
        researchRequired,
        affordableStarSpecialists,
        affordableCarrierSpecialists,
        upgradeCosts: {
            economy: ecoCosts,
            industry: indCosts,
            science: sciCosts,
            warpGate: wgCosts,
        },
        ownCarriers,
        incomingAttacks: [...attackMap.values()],
    };
}
