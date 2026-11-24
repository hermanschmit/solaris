import fs from "fs";
import path from "path";
import { DatabaseSync } from "node:sqlite";
import { SeatMemory } from "../../fog";
import { HumanState, PersistedGameState, PersistedSeatMemory } from "./types";

export function createEmptyHumanState(): HumanState {
    return {
        lastKnownTechLevels: {},
        lastAiResearch: { now: null, next: null, tick: -1 },
        humanResearchPin: null,
        aiCommittedCarriers: {},
        humanPinnedCarriers: {},
        lastValidDirectives: null,
        lastNotesRaw: null,
        lastDiplomaticTiers: {},
        processedEventIds: [],
        dispatchedIdempotencyKeys: [],
        auditLog: [],
        alerts: [],
    };
}

export function createEmptyPersistedState(
    gameId: string,
    playerId: string,
): PersistedGameState {
    return {
        version: 1,
        gameId,
        playerId,
        lastProcessedTick: -1,
        backfilledUpToTick: -1,
        seatMemory: null,
        beliefTracker: null,
        human: createEmptyHumanState(),
    };
}

export function serializeSeatMemory(mem: SeatMemory): PersistedSeatMemory {
    return {
        starsCount: mem.ships.length,
        seenTick: Array.from(mem.seenTick),
        ships: Array.from(mem.ships),
        infrastructure: Array.from(mem.infrastructure),
        warpGate: Array.from(mem.warpGate),
        specialist: Array.from(mem.specialist),
    };
}

export function deserializeSeatMemory(
    saved: PersistedSeatMemory | null | undefined,
    starsCount: number,
): SeatMemory {
    const mem = new SeatMemory(starsCount);
    if (!saved || saved.starsCount !== starsCount) {
        return mem;
    }
    mem.seenTick.set(saved.seenTick.slice(0, starsCount));
    mem.ships.set(saved.ships.slice(0, starsCount));
    mem.infrastructure.set(saved.infrastructure.slice(0, starsCount * 3));
    mem.warpGate.set(saved.warpGate.slice(0, starsCount));
    mem.specialist.set(saved.specialist.slice(0, starsCount));
    return mem;
}

const isSqlitePath = (filePath: string): boolean => {
    const lower = filePath.toLowerCase();
    return (
        lower.endsWith(".db") ||
        lower.endsWith(".sqlite") ||
        lower.endsWith(".sqlite3")
    );
};

/**
 * Per-game persistent state store supporting BOTH SQLite (`.db` / `.sqlite`
 * via Node 24's built-in `node:sqlite`) AND atomic JSON files (`.json`).
 */
export class LiveStateStore {
    readonly filePath: string;
    private sqliteDb: DatabaseSync | null = null;

    constructor(filePath: string) {
        this.filePath = path.resolve(filePath);
        fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
        if (isSqlitePath(this.filePath)) {
            this.sqliteDb = new DatabaseSync(this.filePath);
            this.sqliteDb.exec(`
                CREATE TABLE IF NOT EXISTS admiral_live_state (
                    game_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (game_id, player_id)
                );
            `);
        }
    }

    load(gameId: string, playerId: string): PersistedGameState {
        if (this.sqliteDb) {
            const stmt = this.sqliteDb.prepare(
                "SELECT payload FROM admiral_live_state WHERE game_id = ? AND player_id = ?",
            );
            const row = stmt.get(gameId, playerId) as
                { payload: string } | undefined;
            if (!row?.payload) {
                return createEmptyPersistedState(gameId, playerId);
            }
            const parsed = JSON.parse(row.payload) as PersistedGameState;
            parsed.human = { ...createEmptyHumanState(), ...parsed.human };
            return parsed;
        }

        if (!fs.existsSync(this.filePath)) {
            return createEmptyPersistedState(gameId, playerId);
        }
        const raw = fs.readFileSync(this.filePath, "utf8");
        const parsed = JSON.parse(raw) as PersistedGameState;
        if (parsed.gameId !== gameId || parsed.playerId !== playerId) {
            return createEmptyPersistedState(gameId, playerId);
        }
        parsed.human = { ...createEmptyHumanState(), ...parsed.human };
        return parsed;
    }

    save(state: PersistedGameState) {
        // Bound auditLog, alerts, and idempotency history so multi-week games stay compact.
        if (state.human.auditLog.length > 200) {
            state.human.auditLog = state.human.auditLog.slice(-200);
        }
        if (state.human.alerts.length > 200) {
            state.human.alerts = state.human.alerts.slice(-200);
        }
        if (state.human.dispatchedIdempotencyKeys.length > 500) {
            state.human.dispatchedIdempotencyKeys =
                state.human.dispatchedIdempotencyKeys.slice(-500);
        }

        const payload = JSON.stringify(state, null, 2);

        if (this.sqliteDb) {
            const stmt = this.sqliteDb.prepare(`
                INSERT INTO admiral_live_state (game_id, player_id, updated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(game_id, player_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
            `);
            stmt.run(
                state.gameId,
                state.playerId,
                new Date().toISOString(),
                payload,
            );
            return;
        }

        const tmp = `${this.filePath}.tmp.${process.pid}`;
        fs.writeFileSync(tmp, payload, "utf8");
        fs.renameSync(tmp, this.filePath);
    }

    close() {
        if (this.sqliteDb) {
            this.sqliteDb.close();
            this.sqliteDb = null;
        }
    }
}
