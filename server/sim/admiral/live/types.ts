import { InfrastructureType } from "../../../services/types/Star";

export type OperatingMode = "paused" | "advisor" | "approval" | "autonomous";

export type DiplomaticTier = "ALLIED" | "PROTECTED" | "HOSTILE";

export type NeutralBorderPosture = "defend" | "open";

export interface NotesDirectives {
    mode?: OperatingMode;
    cashReserve?: number;
    attack: string[];
    expectTech: string[];
    neutralBorder?: NeutralBorderPosture;
}

export interface NotesParseResult {
    ok: boolean;
    hasBlock: boolean;
    directives: NotesDirectives;
    warnings: string[];
    fallbackUsed: boolean;
}

export interface WaypointSpec {
    source: string;
    destination: string;
    action:
        | "nothing"
        | "collectAll"
        | "dropAll"
        | "collect"
        | "drop"
        | "collectAllBut"
        | "dropAllBut"
        | "garrison";
    actionShips: number;
    delayTicks: number;
}

export type LiveAction =
    | {
          type: "SetResearchNow";
          preference: string;
          reason?: string;
      }
    | {
          type: "SetResearchQueue";
          queue: string[];
          reason?: string;
      }
    | {
          type: "HireStarSpecialist";
          starId: string;
          starName?: string;
          specialistId: number;
      }
    | {
          type: "UpgradeStar";
          starId: string;
          starName?: string;
          infrastructure: InfrastructureType;
          expectedCost?: number;
      }
    | {
          type: "BuildWarpGate";
          starId: string;
          starName?: string;
          expectedCost?: number;
      }
    | {
          type: "BuildCarrier";
          tempCarrierId: string;
          starId: string;
          starName?: string;
          ships: number;
      }
    | {
          type: "TransferShips";
          carrierId: string;
          starId: string;
          carrierShips: number;
          starShips: number;
      }
    | {
          type: "HireCarrierSpecialist";
          carrierId: string;
          specialistId: number;
      }
    | {
          type: "SaveWaypoints";
          carrierId: string;
          waypoints: WaypointSpec[];
          looped: boolean;
          reason?: string;
          isDefense?: boolean;
      };

export type LiveAlertCode =
    | "SURPRISE_ATTACK"
    | "UNRECALLABLE_FLEET"
    | "PEACE_WAYPOINTS_CANCELLED"
    | "TECH_RECEIVED"
    | "TRADE_OPPORTUNITY"
    | "HUMAN_RESEARCH_PIN"
    | "NOTES_PARSE_WARNING"
    | "ALLY_IN_ATTACK_IGNORED"
    | "STALE_STATE_CONFLICT"
    | "TICK_DEADLINE_GUARD"
    | "AUTH_EXPIRED"
    | "EMERGENCY_RESERVE_DIP"
    | "CREDITS_RECEIVED"
    | "CHAT_MESSAGE_NOTIFICATION";

export interface LiveAlert {
    timestamp: string;
    tick: number;
    severity: "info" | "warn" | "urgent";
    code: LiveAlertCode;
    message: string;
    details?: Record<string, any>;
}

export interface AuditEntry {
    timestamp: string;
    tick: number;
    mode: OperatingMode;
    trigger: string;
    plannedActions: LiveAction[];
    dispatchedCount: number;
    remappedCarriers?: Record<string, string>;
    conflictError?: string;
}

export interface CarrierCommitSnapshot {
    carrierId: string;
    ships: number;
    orbiting: string | null;
    waypoints: WaypointSpec[];
    waypointsHash: string;
    isGift: boolean;
    looped: boolean;
    specialistId: number | null;
    tick: number;
}

export interface PinnedCarrierState {
    carrierId: string;
    reason:
        "gift" | "looped" | "manual_prefix" | "gui_created" | "gui_modified";
    pinnedAtTick: number;
    idleGraceRemaining: number;
}

export interface PersistedSeatMemory {
    starsCount: number;
    seenTick: number[];
    ships: number[];
    infrastructure: number[];
    warpGate: number[];
    specialist: number[];
}

export interface PersistedBeliefTrackerState {
    seat: number;
    lastTick: number;
    history: [number, number[][], number[][]][];
    ghosts: [number, number, number, number][];
}

export interface HumanResearchPin {
    tech: string;
    targetLevel: number;
    pinnedAtTick: number;
}

export interface HumanState {
    lastKnownTechLevels: Record<string, number>;
    lastAiResearch: {
        now: string | null;
        next: string | null;
        queue?: string[] | null;
        tick: number;
    };
    humanResearchPin: HumanResearchPin | null;
    aiCommittedCarriers: Record<string, CarrierCommitSnapshot>;
    humanPinnedCarriers: Record<string, PinnedCarrierState>;
    lastValidDirectives: NotesDirectives | null;
    lastNotesRaw: string | null;
    lastDiplomaticTiers: Record<string, DiplomaticTier>;
    processedEventIds: string[];
    dispatchedIdempotencyKeys: string[];
    auditLog: AuditEntry[];
    alerts: LiveAlert[];
}

export interface PersistedGameState {
    version: 1;
    gameId: string;
    playerId: string;
    lastProcessedTick: number;
    backfilledUpToTick: number;
    seatMemory: PersistedSeatMemory | null;
    beliefTracker: PersistedBeliefTrackerState | null;
    human: HumanState;
}

export interface LiveDaemonConfig {
    apiUrl: string;
    gameId: string;
    email?: string;
    password?: string;
    cookie?: string;
    mode: OperatingMode;
    cashReserveCredits: number;
    neutralBorderPosture: NeutralBorderPosture;
    attackOverrides?: string[];
    expectTech?: string[];
    maxRequestsPerSecond: number;
    carrierGraceTicks: number;
    allowEmergencyDefenseReserveDip: boolean;
    tickOffsetMs: number;
    pollIntervalMs: number;
    deadlineSafetyMs: number;
    statePath: string;
    checkpointPath?: string;
    useSearchPlanner?: boolean;
    searchPlans?: number;
    searchWorlds?: number;
    backfillMaxTicks?: number;
    enableSockets?: boolean;
    approvalHandler?: (actions: LiveAction[], tick: number) => Promise<boolean>;
    onAlert?: (alert: LiveAlert) => void;
}
