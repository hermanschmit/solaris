import { io, Socket } from "socket.io-client";
import { DiplomaticStatus } from "@solaris/common";
import { InfrastructureType } from "../../../services/types/Star";
import { WaypointSpec } from "./types";

export class SolarisHttpError extends Error {
    readonly status: number;
    readonly method: string;
    readonly path: string;
    readonly bodyText: string;

    constructor(
        method: string,
        path: string,
        status: number,
        bodyText: string,
    ) {
        super(`Solaris API ${method} ${path} failed (${status}): ${bodyText}`);
        this.name = "SolarisHttpError";
        this.status = status;
        this.method = method;
        this.path = path;
        this.bodyText = bodyText;
    }
}

export interface SolarisSocketCallbacks {
    onDiplomaticStatusChanged?: (payload: {
        diplomaticStatus: DiplomaticStatus<string>;
    }) => void;
    onTechnologyReceived?: (payload: {
        playerId: string;
        type: string;
        date: string | Date;
        data: {
            fromPlayerId: string;
            toPlayerId: string;
            technology: { name: string; level: number };
        };
    }) => void;
    onCreditsReceived?: (payload: {
        playerId: string;
        data: { fromPlayerId: string; toPlayerId: string; credits: number };
    }) => void;
    onCreditsSpecialistsReceived?: (payload: {
        playerId: string;
        data: {
            fromPlayerId: string;
            toPlayerId: string;
            creditsSpecialists: number;
        };
    }) => void;
    onGameMessageSent?: (payload: any) => void;
    onGameStarted?: (payload: any) => void;
    onAuthExpired?: () => void;
}

/**
 * Live Solaris HTTP REST + Socket.IO client.
 *
 * Enforces:
 * - Independent session login (`POST /api/auth/login`) so human browser tabs
 *   on the same account are NEVER logged out.
 * - Strict sequential request serialization (`concurrency = 1`) with a
 *   configurable rate cap (default 3 req/s; 2 req/s for Two-Hybrid shared-IP testing)
 *   staying safely under Solaris's 10 req/s per-IP `express-rate-limit` & `playerMutex`.
 * - Deterministic `Idempotency-Key` headers on all `PUT` mutations.
 * - Zero code paths for `/diplomacy/*` writes, `/trade/*` writes, or `/ready`.
 */
export class SolarisLiveClient {
    readonly apiUrl: string;
    private email?: string;
    private password?: string;
    private cookieHeader: string | null;
    private minIntervalMs: number;
    private lastRequestAt = 0;
    private queue: Promise<any> = Promise.resolve();
    private socket: Socket | null = null;
    private callbacks: SolarisSocketCallbacks = {};

    constructor(options: {
        apiUrl: string;
        email?: string;
        password?: string;
        cookie?: string;
        maxRequestsPerSecond?: number;
    }) {
        this.apiUrl = options.apiUrl.replace(/\/+$/, "");
        this.email = options.email;
        this.password = options.password;
        this.cookieHeader = options.cookie?.trim() || null;
        const rps = Math.max(0.25, options.maxRequestsPerSecond ?? 3);
        this.minIntervalMs = Math.ceil(1000 / rps);
    }

    getCookie(): string | null {
        return this.cookieHeader;
    }

    setCallbacks(callbacks: SolarisSocketCallbacks) {
        this.callbacks = callbacks;
    }

    private async sleep(ms: number) {
        if (ms <= 0) return;
        await new Promise((resolve) => setTimeout(resolve, ms));
    }

    private enqueue<T>(fn: () => Promise<T>): Promise<T> {
        const run = this.queue.then(async () => {
            const waitMs = this.lastRequestAt + this.minIntervalMs - Date.now();
            if (waitMs > 0) {
                await this.sleep(waitMs);
            }
            this.lastRequestAt = Date.now();
            return fn();
        });
        // Keep queue alive even if a single request rejects.
        this.queue = run.catch(() => {});
        return run;
    }

    async login(): Promise<{ _id?: string; username?: string }> {
        if (this.cookieHeader && (!this.email || !this.password)) {
            return this.verifySession();
        }
        if (!this.email || !this.password) {
            throw new Error(
                "Configure either (email + password) or cookie for SolarisLiveClient.",
            );
        }

        return this.enqueue(async () => {
            const res = await fetch(`${this.apiUrl}/api/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email: this.email,
                    password: this.password,
                }),
            });
            const bodyText = await res.text();
            if (!res.ok) {
                throw new SolarisHttpError(
                    "POST",
                    "/api/auth/login",
                    res.status,
                    bodyText,
                );
            }
            const setCookies = res.headers
                .getSetCookie()
                .map((c) => c.split(";")[0]);
            if (setCookies.length > 0) {
                this.cookieHeader = setCookies.join("; ");
            }
            return this.parseResponseText(bodyText);
        });
    }

    private parseResponseText<T = any>(text: string): T {
        const trimmed = text.trim();
        if (!trimmed) return {} as T;
        try {
            return JSON.parse(trimmed) as T;
        } catch {
            return {} as T;
        }
    }

    async verifySession(): Promise<{ _id: string; username: string }> {
        return this.request("POST", "/api/auth/verify", {});
    }

    private async request<T = any>(
        method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
        path: string,
        body?: any,
        idempotencyKey?: string,
        attempt = 0,
    ): Promise<T> {
        if (!this.cookieHeader) {
            await this.login();
        }

        return this.enqueue(async () => {
            const headers: Record<string, string> = {};
            if (this.cookieHeader) {
                headers["Cookie"] = this.cookieHeader;
            }
            if (body !== undefined) {
                headers["Content-Type"] = "application/json";
            }
            if (idempotencyKey) {
                headers["Idempotency-Key"] = idempotencyKey;
            }

            const res = await fetch(`${this.apiUrl}${path}`, {
                method,
                headers,
                body: body !== undefined ? JSON.stringify(body) : undefined,
            });
            const text = await res.text();

            if (res.status === 429 && attempt < 3) {
                // Respect rate-limit backoff and retry safely (idempotent on PUTs).
                await this.sleep(1100 * (attempt + 1));
                this.lastRequestAt = Date.now();
                const retryRes = await fetch(`${this.apiUrl}${path}`, {
                    method,
                    headers,
                    body: body !== undefined ? JSON.stringify(body) : undefined,
                });
                const retryText = await retryRes.text();
                if (!retryRes.ok) {
                    throw new SolarisHttpError(
                        method,
                        path,
                        retryRes.status,
                        retryText,
                    );
                }
                return this.parseResponseText<T>(retryText);
            }

            if (res.status === 401) {
                this.callbacks.onAuthExpired?.();
            }

            if (!res.ok) {
                throw new SolarisHttpError(method, path, res.status, text);
            }
            return this.parseResponseText<T>(text);
        });
    }

    // ---- Read-Only Synchronization Endpoints ----

    async getState(gameId: string): Promise<any> {
        return this.request("GET", `/api/game/${gameId}/state`);
    }

    async getGalaxy(gameId: string, tick?: number): Promise<any> {
        const qs =
            tick != null ? `?tick=${encodeURIComponent(String(tick))}` : "";
        return this.request("GET", `/api/game/${gameId}/galaxy${qs}`);
    }

    async getDiplomacy(gameId: string): Promise<DiplomaticStatus<string>[]> {
        return this.request("GET", `/api/game/${gameId}/diplomacy`);
    }

    async getEvents(
        gameId: string,
        page = 0,
        pageSize = 30,
        category = "trade",
    ): Promise<{ count: number; events: any[] }> {
        const qs = `?page=${page}&pageSize=${pageSize}&category=${encodeURIComponent(category)}`;
        return this.request("GET", `/api/game/${gameId}/events${qs}`);
    }

    async getNotes(gameId: string): Promise<{ notes?: string }> {
        return this.request("GET", `/api/game/${gameId}/notes`);
    }

    async getMonthlySpecialistBans(): Promise<{
        carrier: { id: number }[];
        star: { id: number }[];
    }> {
        return this.request("GET", `/api/game/specialists/bans`);
    }

    // ---- Tactical & Research Write Endpoints (with Idempotency-Key) ----

    async updateResearchNow(
        gameId: string,
        preference: string,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/research/now`,
            { preference },
            idempotencyKey,
        );
    }

    async updateResearchNext(
        gameId: string,
        preference: string,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/research/next`,
            { preference },
            idempotencyKey,
        );
    }

    async hireStarSpecialist(
        gameId: string,
        starId: string,
        specialistId: number,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/star/${starId}/hire/${specialistId}`,
            {},
            idempotencyKey,
        );
    }

    async upgradeStar(
        gameId: string,
        starId: string,
        infrastructure: InfrastructureType,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/star/upgrade/${infrastructure}`,
            { starId },
            idempotencyKey,
        );
    }

    async buildWarpGate(
        gameId: string,
        starId: string,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/star/build/warpgate`,
            { starId },
            idempotencyKey,
        );
    }

    async buildCarrier(
        gameId: string,
        starId: string,
        ships: number,
        idempotencyKey: string,
    ): Promise<{ carrier: { _id: string }; starShips: number }> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/star/build/carrier`,
            { starId, ships },
            idempotencyKey,
        );
    }

    async transferShips(
        gameId: string,
        carrierId: string,
        starId: string,
        carrierShips: number,
        starShips: number,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/carrier/${carrierId}/transfer`,
            { starId, carrierShips, starShips },
            idempotencyKey,
        );
    }

    async hireCarrierSpecialist(
        gameId: string,
        carrierId: string,
        specialistId: number,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/carrier/${carrierId}/hire/${specialistId}`,
            {},
            idempotencyKey,
        );
    }

    async saveWaypoints(
        gameId: string,
        carrierId: string,
        waypoints: WaypointSpec[],
        looped: boolean,
        idempotencyKey: string,
    ): Promise<any> {
        return this.request(
            "PUT",
            `/api/game/${gameId}/carrier/${carrierId}/waypoints`,
            { waypoints, looped },
            idempotencyKey,
        );
    }

    // ---- Real-Time Socket.IO Subscription ----

    connectSocket(gameId: string, playerId: string): Promise<void> {
        this.disconnectSocket();
        if (!this.cookieHeader) {
            throw new Error("Must authenticate before connecting Socket.IO.");
        }

        return new Promise((resolve) => {
            let settled = false;
            const socket = io(this.apiUrl, {
                transports: ["websocket", "polling"],
                extraHeaders: {
                    Cookie: this.cookieHeader!,
                },
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
            });
            this.socket = socket;

            socket.on("connect", () => {
                socket.emit("userJoined", {});
                socket.emit("gameRoomJoined", { gameId, playerId });
                if (!settled) {
                    settled = true;
                    setTimeout(resolve, 60);
                }
            });

            socket.on("connect_error", () => {
                if (!settled) {
                    settled = true;
                    resolve();
                }
            });

            socket.on("playerDiplomaticStatusChanged", (data) => {
                this.callbacks.onDiplomaticStatusChanged?.(data);
            });
            socket.on("playerTechnologyReceived", (data) => {
                this.callbacks.onTechnologyReceived?.(data);
            });
            socket.on("playerCreditsReceived", (data) => {
                this.callbacks.onCreditsReceived?.(data);
            });
            socket.on("playerCreditsSpecialistsReceived", (data) => {
                this.callbacks.onCreditsSpecialistsReceived?.(data);
            });
            socket.on("gameMessageSent", (data) => {
                this.callbacks.onGameMessageSent?.(data);
            });
            socket.on("gameStarted", (data) => {
                this.callbacks.onGameStarted?.(data);
            });
        });
    }

    disconnectSocket() {
        if (this.socket) {
            this.socket.removeAllListeners();
            this.socket.disconnect();
            this.socket = null;
        }
    }
}
