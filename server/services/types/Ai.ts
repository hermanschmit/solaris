export interface KnownAttack {
    arrivalTick: number;
    starId: string;
    carriersOnTheWay: string[];
}

export interface InvasionInProgress {
    arrivalTick: number;
    star: string;
}

export interface AiState {
    knownAttacks: KnownAttack[];
    invasionsInProgress: InvasionInProgress[];
    startedClaims: string[];
}

// Tuning for the advanced AI. Every field has the value the AI has always
// used, so the defaults reproduce its original behavior.
export interface AiConfig {
    // Share of credits (in percent) bulk-upgraded into science and industry
    // on the first tick of each cycle.
    firstTickSciencePercentage: number;
    firstTickIndustryPercentage: number;
    // Share of the remaining credits bulk-upgraded into economy on the last
    // tick of each cycle.
    lastTickEconomyPercentage: number;
    // Ships sent to invade, as a multiple of the ships needed to win.
    invasionAttackFactor: number;
    // Angular gap to other own stars that makes a star a border star.
    borderStarAngleThresholdDegrees: number;
    // Ships stockpiled at a star, in cycles of its production, before a new
    // carrier is bought to move them to the border.
    logisticStockpileCycles: number;
    // Credits kept back before buying a carrier for anything but defense,
    // as a multiple of total economy.
    carrierReserveEconomyMultiplier: number;
    // Maximum travel time for invasions and claims, in cycles.
    orderReachCycles: number;
    // Weights for scoring a star by its infrastructure.
    starValueWeights: {
        economy: number;
        industry: number;
        science: number;
    };
    // Orders are processed by category first, highest priority first.
    orderPriorities: {
        defend: number;
        invade: number;
        claim: number;
    };
}

export const DEFAULT_AI_CONFIG: AiConfig = {
    firstTickSciencePercentage: 20,
    firstTickIndustryPercentage: 30,
    lastTickEconomyPercentage: 100,
    invasionAttackFactor: 1.5,
    borderStarAngleThresholdDegrees: 120,
    logisticStockpileCycles: 0.7,
    carrierReserveEconomyMultiplier: 5,
    orderReachCycles: 2,
    starValueWeights: {
        economy: 1,
        industry: 2,
        science: 3,
    },
    orderPriorities: {
        defend: 4,
        invade: 3,
        claim: 2,
    },
};
