import { SimEngine } from "../../engine";
import { Game } from "../../../services/types/Game";
import { DiplomaticPolicy } from "./diplomaticPolicy";
import { HumanState, LiveAction, LiveAlert } from "./types";

export interface ReceivedTechRecord {
    technology: string;
    level: number;
    fromPlayerId?: string;
    source: "socket" | "event" | "level_diff";
}

export interface ResearchMaskPlan {
    // Length 1 + T: index 0 = keep current, 1..T = technologies[ti]
    allowedMask: boolean[];
    allyHeldTechs: {
        tech: string;
        allyAlias: string;
        allyLevel: number;
        myLevel: number;
        sendCostCredits: number;
    }[];
}

export interface ResearchCoordinationResult {
    desiredNow: string;
    desiredNext: string;
    actions: LiveAction[];
    receivedTechs: ReceivedTechRecord[];
    humanPinned: boolean;
    sunkCostProtected: boolean;
}

/**
 * Detects technologies received from other players via:
 * 1. Socket.IO `playerTechnologyReceived` notifications,
 * 2. Trade events (`GET /api/game/:gameId/events?category=trade`), and
 * 3. Tech-level diff fallback (`me.research[t].level > lastKnownTechLevels[t]`).
 */
export function detectReceivedTechnologies(params: {
    me: any;
    technologies: string[];
    human: HumanState;
    socketTechEvents?: {
        technology: string;
        level: number;
        fromPlayerId?: string;
    }[];
    tradeEvents?: any[];
    allRecentEvents?: any[];
}): ReceivedTechRecord[] {
    const { me, technologies, human } = params;
    const found = new Map<string, ReceivedTechRecord>();
    const processedSet = new Set(human.processedEventIds);

    for (const sEvt of params.socketTechEvents ?? []) {
        if (sEvt?.technology) {
            found.set(sEvt.technology, {
                technology: sEvt.technology,
                level: sEvt.level,
                fromPlayerId: sEvt.fromPlayerId,
                source: "socket",
            });
        }
    }

    const naturalFinishTechs = new Set<string>();
    for (const evt of params.allRecentEvents ?? []) {
        const eid = evt?._id?.toString();
        if (eid && processedSet.has(eid)) continue;
        if (
            evt?.type === "playerResearchComplete" ||
            evt?.type === "playerExperimentation"
        ) {
            const techKey = evt?.data?.technologyKey ?? evt?.data?.technology;
            if (typeof techKey === "string") naturalFinishTechs.add(techKey);
        }
    }

    for (const evt of params.tradeEvents ?? []) {
        const eid = evt?._id?.toString();
        if (eid && processedSet.has(eid)) continue;
        if (eid) {
            human.processedEventIds.push(eid);
            processedSet.add(eid);
        }
        if (evt?.type === "playerTechnologyReceived") {
            const techObj = evt?.data?.technology;
            const name = techObj?.name ?? techObj?.technology;
            const level = Number(techObj?.level ?? 0);
            if (typeof name === "string" && !found.has(name)) {
                found.set(name, {
                    technology: name,
                    level,
                    fromPlayerId: evt?.data?.fromPlayerId?.toString(),
                    source: "event",
                });
            }
        }
    }

    // Channel 3: Tech-level diff fallback vs persisted `lastKnownTechLevels`.
    if (Object.keys(human.lastKnownTechLevels).length > 0) {
        for (const t of technologies) {
            const prevLvl = human.lastKnownTechLevels[t];
            const curLvl = Number(me.research?.[t]?.level ?? 0);
            const curProg = Number(me.research?.[t]?.progress ?? 0);
            if (
                prevLvl != null &&
                curLvl > prevLvl &&
                !found.has(t) &&
                !naturalFinishTechs.has(t)
            ) {
                // TradeService.sendTechnology resets progress to 0!
                if (curProg === 0 || t === me.researchingNow) {
                    found.set(t, {
                        technology: t,
                        level: curLvl,
                        source: "level_diff",
                    });
                }
            }
        }
    }

    if (human.processedEventIds.length > 400) {
        human.processedEventIds = human.processedEventIds.slice(-250);
    }

    return [...found.values()];
}

/**
 * Computes the eligible research mask for `AdmiralNet` (`[1 + T]`), masking out
 * unresearchable techs, `expect_tech` hints, and technologies an in-range
 * formal ally already holds at a higher level (provided at least one other
 * technology remains eligible).
 */
export function computeResearchMasks(params: {
    engine: SimEngine;
    game: Game;
    me: any;
    technologies: string[];
    diplomacy: DiplomaticPolicy;
    expectTech?: string[];
}): ResearchMaskPlan {
    const { engine, game, me, technologies, diplomacy } = params;
    const researchable = technologies.map((t) =>
        engine.container.technologyService.isTechnologyResearchable(
            game,
            t as any,
        ),
    );
    const expectSet = new Set(
        (params.expectTech ?? []).map((s) => s.toLowerCase()),
    );
    const tradeCostPerLvl = Number(game.settings.player?.tradeCost ?? 15);
    const tradeScanningOnly = game.settings.player?.tradeScanning === "scanned";

    const allyHeldByTech = new Map<
        string,
        {
            tech: string;
            allyAlias: string;
            allyLevel: number;
            myLevel: number;
            sendCostCredits: number;
        }
    >();

    for (const p of game.galaxy.players) {
        const pid = p._id.toString();
        if (pid === me._id.toString() || p.defeated) continue;
        if (!diplomacy.isAllied(pid)) continue;
        // In live `/galaxy`, `isInScanningRange` is populated on players; in sim
        // fall back to scanningService if undefined.
        const inTradeRange =
            !tradeScanningOnly ||
            (p as any).isInScanningRange === true ||
            ((p as any).isInScanningRange === undefined &&
                engine.container.scanningService.isInScanningRangeOfPlayer(
                    game,
                    me,
                    p,
                ));
        if (!inTradeRange) continue;

        for (const t of technologies) {
            const myLvl = Number(me.research?.[t]?.level ?? 1);
            const allyLvl = Number((p.research as any)?.[t]?.level ?? 1);
            if (allyLvl > myLvl) {
                const prev = allyHeldByTech.get(t);
                if (!prev || allyLvl > prev.allyLevel) {
                    allyHeldByTech.set(t, {
                        tech: t,
                        allyAlias: p.alias,
                        allyLevel: allyLvl,
                        myLevel: myLvl,
                        sendCostCredits: (myLvl + 1) * tradeCostPerLvl,
                    });
                }
            }
        }
    }

    // Apply progressive filtering so at least 1 researchable tech always remains unmasked.
    let mask = researchable.slice();

    const withoutExpected = mask.map(
        (ok, i) => ok && !expectSet.has(technologies[i].toLowerCase()),
    );
    if (withoutExpected.some(Boolean)) {
        mask = withoutExpected;
    }

    const withoutAllyHeld = mask.map(
        (ok, i) => ok && !allyHeldByTech.has(technologies[i]),
    );
    if (withoutAllyHeld.some(Boolean)) {
        mask = withoutAllyHeld;
    }

    return {
        allowedMask: [true, ...mask],
        allyHeldTechs: [...allyHeldByTech.values()],
    };
}

/**
 * Coordinates `researchingNow` and `researchingNext` across:
 * 1. Manual GUI research pin detection (`me.researchingNow !== lastAiResearch.now`),
 * 2. Instant reaction to received technologies (invalidating stale pins & progress wipes),
 * 3. `expect_tech` & ally already-holds deprioritization (+ trade-opportunity alerts),
 * 4. >=50% sunk-cost protection on `researchingNow`, and
 * 5. Policy `res_logits` ranking (#1 -> `researchingNow`, #2 -> `researchingNext`, never `"random"`).
 */
export function coordinateResearch(params: {
    engine: SimEngine;
    game: Game;
    me: any;
    technologies: string[];
    human: HumanState;
    maskPlan: ResearchMaskPlan;
    receivedTechs: ReceivedTechRecord[];
    resLogits?: number[] | null;
    policyChoice?: string | null;
    expectTech?: string[];
    isCycleStart: boolean;
    forceReevaluate?: boolean;
    onAlert?: (alert: LiveAlert) => void;
}): ResearchCoordinationResult {
    const { engine, game, me, technologies, human, maskPlan, receivedTechs } =
        params;
    const tick = game.state.tick;
    const receivedMap = new Map(receivedTechs.map((r) => [r.technology, r]));
    const expectSet = new Set(
        (params.expectTech ?? []).map((s) => s.toLowerCase()),
    );

    // Emit alerts for newly received technologies.
    for (const rec of receivedTechs) {
        params.onAlert?.({
            timestamp: new Date().toISOString(),
            tick,
            severity: "info",
            code: "TECH_RECEIVED",
            message: `Received technology '${rec.technology}' (level ${rec.level}) via ${rec.source}; re-evaluating research queue immediately.`,
            details: rec,
        });
    }

    const researchable = technologies.filter((t) =>
        engine.container.technologyService.isTechnologyResearchable(
            game,
            t as any,
        ),
    );
    if (researchable.length === 0) {
        return {
            desiredNow: me.researchingNow,
            desiredNext: me.researchingNext,
            actions: [],
            receivedTechs,
            humanPinned: false,
            sunkCostProtected: false,
        };
    }

    const curNow: string = me.researchingNow;
    const curNext: string = me.researchingNext;

    // 1. Update / detect Manual GUI Research Pin!
    if (
        human.lastAiResearch.now != null &&
        curNow !== human.lastAiResearch.now &&
        !receivedMap.has(curNow)
    ) {
        const curLvl = Number(me.research?.[curNow]?.level ?? 1);
        if (human.humanResearchPin?.tech !== curNow) {
            human.humanResearchPin = {
                tech: curNow,
                targetLevel: curLvl + 1,
                pinnedAtTick: tick,
            };
            params.onAlert?.({
                timestamp: new Date().toISOString(),
                tick,
                severity: "info",
                code: "HUMAN_RESEARCH_PIN",
                message: `Detected manual GUI research selection '${curNow}' (pinned until level ${curLvl + 1} completes or tech is received).`,
                details: human.humanResearchPin,
            });
        }
    }

    // Expire or invalidate Human Research Pin if completed or received via trade.
    if (human.humanResearchPin) {
        const pin = human.humanResearchPin;
        const pinCurLvl = Number(me.research?.[pin.tech]?.level ?? 1);
        if (
            receivedMap.has(pin.tech) ||
            pinCurLvl >= pin.targetLevel ||
            !researchable.includes(pin.tech)
        ) {
            human.humanResearchPin = null;
        }
    }

    // Score every researchable technology using `AdmiralNet`'s `res_logits[1 + ti]`
    // (or a balanced level/prior fallback when logits are absent), penalizing
    // `expect_tech` and ally-held technologies unless nothing else is available.
    const allyHeldMap = new Map(maskPlan.allyHeldTechs.map((a) => [a.tech, a]));
    const defaultOrder = [
        "weapons",
        "manufacturing",
        "hyperspace",
        "terraforming",
        "banking",
        "scanning",
        "experimentation",
        "specialists",
    ];

    const scoreTech = (t: string): number => {
        const ti = technologies.indexOf(t);
        let score: number;
        if (params.resLogits && params.resLogits.length > 1 + ti && ti >= 0) {
            const raw = params.resLogits[1 + ti];
            score = Number.isFinite(raw) && raw > -1e7 ? raw : -50;
        } else {
            const lvl = Number(me.research?.[t]?.level ?? 1);
            const prioIdx = defaultOrder.indexOf(t);
            const base = prioIdx >= 0 ? 10 - prioIdx * 0.8 : 2;
            score = base - lvl * 0.9;
            if (params.policyChoice === t) score += 20;
        }
        if (expectSet.has(t.toLowerCase())) score -= 200;
        if (allyHeldMap.has(t)) score -= 100;
        return score;
    };

    const ranked = [...researchable].sort(
        (a, b) => scoreTech(b) - scoreTech(a),
    );

    const pickBestNext = (chosenNow: string): string => {
        // Respect manual GUI `researchingNext` if human explicitly set a non-random
        // next tech that isn't equal to `chosenNow`, expected, or ally-held.
        if (
            human.lastAiResearch.next != null &&
            curNext !== human.lastAiResearch.next &&
            curNext !== "random" &&
            curNext !== chosenNow &&
            researchable.includes(curNext) &&
            !expectSet.has(curNext.toLowerCase())
        ) {
            return curNext;
        }
        const second = ranked.find((t) => t !== chosenNow);
        return second ?? chosenNow;
    };

    let desiredNow = curNow;
    let humanPinned = false;
    let sunkCostProtected = false;

    const shouldEvaluateNow =
        params.isCycleStart ||
        params.forceReevaluate ||
        receivedTechs.length > 0 ||
        human.lastAiResearch.now == null ||
        curNext === "random";

    if (human.humanResearchPin) {
        desiredNow = human.humanResearchPin.tech;
        humanPinned = true;
    } else if (shouldEvaluateNow) {
        const rps = engine.container.researchService.researchProgressService;
        const curLvl = Number(me.research?.[curNow]?.level ?? 1);
        const curProg = Number(me.research?.[curNow]?.progress ?? 0);
        const curReq = Number(
            rps.getRequiredResearchProgress(game, curNow as any, curLvl) ?? 0,
        );
        const frac = curReq > 0 ? curProg / curReq : 0;

        if (
            !receivedMap.has(curNow) &&
            !expectSet.has(curNow.toLowerCase()) &&
            researchable.includes(curNow) &&
            frac >= 0.5
        ) {
            desiredNow = curNow;
            sunkCostProtected = true;
        } else {
            desiredNow = ranked[0];
            // Surface trade-opportunity tip if an ally holds a higher level of a skipped tech!
            for (const allyTip of maskPlan.allyHeldTechs) {
                if (allyTip.tech !== desiredNow) {
                    params.onAlert?.({
                        timestamp: new Date().toISOString(),
                        tick,
                        severity: "info",
                        code: "TRADE_OPPORTUNITY",
                        message: `Ally '${allyTip.allyAlias}' already has ${allyTip.tech} ${allyTip.allyLevel} (cost ${allyTip.sendCostCredits}cr to send level ${allyTip.myLevel + 1}) — researching '${desiredNow}' instead.`,
                        details: allyTip,
                    });
                }
            }
        }
    }

    const desiredNext = pickBestNext(desiredNow);

    // Update known tech levels snapshot for future tick diff detection.
    for (const t of technologies) {
        human.lastKnownTechLevels[t] = Number(me.research?.[t]?.level ?? 1);
    }

    const actions: LiveAction[] = [];
    if (desiredNow !== curNow) {
        actions.push({
            type: "SetResearchNow",
            preference: desiredNow,
            reason: receivedTechs.length > 0 ? "tech_received" : "policy",
        });
    }
    if (desiredNext !== curNext) {
        actions.push({
            type: "SetResearchNext",
            preference: desiredNext,
            reason: "policy_next",
        });
    }

    return {
        desiredNow,
        desiredNext,
        actions,
        receivedTechs,
        humanPinned,
        sunkCostProtected,
    };
}
