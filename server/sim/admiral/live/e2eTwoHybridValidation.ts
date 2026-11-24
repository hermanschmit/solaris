import "../../env";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import express from "express";
import { Server as SocketIOServer } from "socket.io";
import { Config } from "../../../config/types/Config";
import mongooseLoader from "../../../db";
import { gameTickJob } from "../../../jobs/jobs/gameTick";
import containerLoader from "../../../services";
import SessionService from "../../../services/session";
import { DependencyContainer } from "../../../services/types/DependencyContainer";
import { objectIdFromString } from "../../../services/types/DBObjectId";
import { logger, setupLogging } from "../../../utils/logging";
import expressLoader from "../../../api/express";
import { buildIdempotencyKey, dispatchLiveActions } from "./commandRecorder";
import { LiveAdmiralDaemon } from "./daemon";
import { snapshotCarrier } from "./sharedSeat";
import { SolarisHttpError } from "./solarisClient";
import { LiveAction, LiveAlert } from "./types";

const MONGO_CONTAINER_NAME = "solaris-admiral-e2e-mongo";
const MONGO_HOST_PORT = 27018;

/**
 * Lightweight simulated Browser GUI client with its own independent cookie jar
 * and pacing so 2 Browser GUI clients + 2 Admiral Daemons share 127.0.0.1
 * against Solaris's unmodified 10 req/s `express-rate-limit`.
 */
class BrowserGuiClient {
    private cookie = "";
    private lastReqAt = 0;

    constructor(
        readonly apiUrl: string,
        private readonly minIntervalMs = 220,
    ) {}

    getCookie(): string {
        return this.cookie;
    }

    private async pace() {
        const wait = this.lastReqAt + this.minIntervalMs - Date.now();
        if (wait > 0) {
            await new Promise((r) => setTimeout(r, wait));
        }
        this.lastReqAt = Date.now();
    }

    async request<T = any>(
        method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
        routePath: string,
        body?: any,
    ): Promise<T> {
        await this.pace();
        const headers: Record<string, string> = {};
        if (this.cookie) headers["Cookie"] = this.cookie;
        if (body !== undefined) headers["Content-Type"] = "application/json";

        let res = await fetch(`${this.apiUrl}${routePath}`, {
            method,
            headers,
            body: body !== undefined ? JSON.stringify(body) : undefined,
        });
        if (res.status === 429) {
            await new Promise((r) => setTimeout(r, 1100));
            this.lastReqAt = Date.now();
            res = await fetch(`${this.apiUrl}${routePath}`, {
                method,
                headers,
                body: body !== undefined ? JSON.stringify(body) : undefined,
            });
        }

        const setCookies = res.headers
            .getSetCookie()
            .map((c) => c.split(";")[0]);
        if (setCookies.length > 0) {
            this.cookie = setCookies.join("; ");
        }

        const text = await res.text();
        if (!res.ok) {
            throw new SolarisHttpError(method, routePath, res.status, text);
        }
        const trimmed = text.trim();
        if (!trimmed) return {} as T;
        try {
            return JSON.parse(trimmed) as T;
        } catch {
            return {} as T;
        }
    }
}

function startMongoDockerContainer(): void {
    spawnSync("docker", ["rm", "-f", MONGO_CONTAINER_NAME], {
        stdio: "ignore",
    });
    execFileSync(
        "docker",
        [
            "run",
            "-d",
            "--rm",
            "--name",
            MONGO_CONTAINER_NAME,
            "-p",
            `127.0.0.1:${MONGO_HOST_PORT}:27017`,
            "-e",
            "GLIBC_TUNABLES=glibc.pthread.rseq=1",
            "mongo:8.0",
        ],
        { stdio: "pipe" },
    );

    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
        const probe = spawnSync(
            "docker",
            [
                "exec",
                MONGO_CONTAINER_NAME,
                "mongosh",
                "--quiet",
                "--eval",
                "db.runCommand({ ping: 1 }).ok",
            ],
            { encoding: "utf8" },
        );
        if (probe.status === 0 && probe.stdout.trim().endsWith("1")) {
            return;
        }
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 300);
    }
    throw new Error("Timed out waiting for mongo:8.0 Docker container.");
}

function stopMongoDockerContainer(): void {
    spawnSync("docker", ["rm", "-f", MONGO_CONTAINER_NAME], {
        stdio: "ignore",
    });
}

function createAdmiralCheckpoint(ckptPath: string): void {
    const pyDir = path.resolve(__dirname, "../python");
    const simPyDir = path.resolve(__dirname, "../../python");
    const script = [
        "import sys, torch",
        `sys.path.insert(0, ${JSON.stringify(simPyDir)})`,
        `sys.path.insert(0, ${JSON.stringify(pyDir)})`,
        "from admiral_net import AdmiralNet",
        "torch.manual_seed(7)",
        "net = AdmiralNet(n_techs=8)",
        `torch.save({'model': net.state_dict(), 'config': {'n_techs': 8}, 'step': 1}, ${JSON.stringify(ckptPath)})`,
    ].join("; ");
    execFileSync("python3", ["-c", script], { stdio: "pipe" });
}

async function waitUntil(
    predicate: () => Promise<boolean>,
    timeoutMs = 5000,
    intervalMs = 200,
): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error(`Condition did not become true within ${timeoutMs}ms.`);
}

export interface TwoHybridValidationSummary {
    gameId: string;
    alphaPlayerId: string;
    betaPlayerId: string;
    scenariosVerified: string[];
}

/**
 * Runs all 8 Two-Hybrid end-to-end verification scenarios from Section 6.2 of
 * `live_diplomatic_agent_design.md` against:
 * - Real Docker `mongo:8.0`,
 * - Real Solaris Express + Socket.IO + SessionStore + 10 req/s `express-rate-limit`
 *   + `idempotencyKeyCache` + `playerMutex` + `gameTickJob` + `historyService`,
 * - Real PyTorch `AdmiralNet` + `BeliefTracker` RPC worker, and
 * - Real SQLite (`node:sqlite`) persistent state stores.
 */
export async function runTwoHybridE2eValidation(): Promise<TwoHybridValidationSummary> {
    const tmpDir = fs.mkdtempSync(
        path.join(os.tmpdir(), "solaris-admiral-two-hybrid-"),
    );
    const ckptPath = path.join(tmpDir, "admiral_e2e.pt");
    const alphaDbPath = path.join(tmpDir, "state_alpha.db");
    const betaDbPath = path.join(tmpDir, "state_beta.db");
    const attachDbPath = path.join(tmpDir, "state_attach.db");

    let mongo: any = null;
    let server: http.Server | null = null;
    let socketServer: SocketIOServer | null = null;
    const activeDaemons: LiveAdmiralDaemon[] = [];
    const scenariosVerified: string[] = [];

    const registerDaemon = (d: LiveAdmiralDaemon) => {
        activeDaemons.push(d);
        return d;
    };

    try {
        console.log(
            "[E2E] 1/10 Starting mongo:8.0 Docker container & saving AdmiralNet checkpoint...",
        );
        startMongoDockerContainer();
        createAdmiralCheckpoint(ckptPath);

        setupLogging();
        const log = logger("admiral-e2e");
        const baseConfig: Config = {
            port: "0",
            http: {
                keepAliveTimeout: 120000,
                headersTimeout: 125000,
            },
            sessionSecret: "admiral-two-hybrid-e2e-secret",
            sessionSecureCookies: false,
            connectionString: `mongodb://127.0.0.1:${MONGO_HOST_PORT}/solaris_admiral_e2e`,
            serverUrl: "http://127.0.0.1",
            clientUrl: "http://127.0.0.1",
            corsUrls: ["http://127.0.0.1"],
            cacheEnabled: false,
            logging: "pretty",
            logLevel: "error",
            smtp: { enabled: false },
            paypal: { environment: "sandbox" },
            discord: {},
            frontend: {
                appSocketsHost: "127.0.0.1",
                appApiHost: "http://127.0.0.1",
                appDocumentationUrl:
                    "https://solaris-games.github.io/solaris-docs/",
                appProviderInformationUrl: "",
                appDiscordOAuthUrl: "",
            },
        };

        mongo = await mongooseLoader(baseConfig, {});
        const app = express();
        server = http.createServer(app);
        socketServer = new SocketIOServer(server, {
            cors: {
                origin: baseConfig.corsUrls,
                methods: ["POST", "PUT", "PATCH", "GET", "DELETE", "OPTIONS"],
                credentials: true,
            },
            transports: ["websocket", "polling"],
        });

        const container: DependencyContainer = containerLoader(
            baseConfig,
            socketServer,
            log,
        );
        const origAiPlay = container.aiService.play.bind(container.aiService);
        container.aiService.play = async (evSvc: any, g: any, p: any) => {
            try {
                await origAiPlay(evSvc, g, p);
            } catch {
                // Guard against upstream Solaris AIService._performLogistics >30-waypoint bug
                // aborting the tick loop before `game.save()`.
            }
        };

        const sessionStore = SessionService.createSessionStore(
            container.sessionRepository,
        );
        container.sessionService.setSessionStorage(sessionStore);
        container.socketService.setSessionStorage(sessionStore);
        await expressLoader(baseConfig, app, container, sessionStore);

        await new Promise<void>((resolve) => {
            server!.listen(0, "127.0.0.1", () => resolve());
        });
        const addr = server.address();
        assert(addr && typeof addr === "object", "Expected bound TCP port");
        const apiUrl = `http://127.0.0.1:${addr.port}`;
        console.log(
            `[E2E] Real Solaris API + Socket.IO listening at ${apiUrl}`,
        );

        const advanceServerTick = async () => {
            const before = await container.gameService.getByIdLean(
                objectIdFromString(gameId),
                { "state.tick": 1 },
            );
            const prevTick = Number(before?.state?.tick ?? 0);
            await container.gameService.gameRepo.updateOne(
                { _id: objectIdFromString(gameId) },
                { $set: { "state.forceTick": true } },
            );
            const ac = new AbortController();
            await gameTickJob(container)(ac.signal);
            const after = await container.gameService.getByIdLean(
                objectIdFromString(gameId),
                { "state.tick": 1 },
            );
            assert.equal(
                Number(after?.state?.tick ?? 0),
                prevTick + 1,
                `Expected server tick to advance from ${prevTick} to ${prevTick + 1}`,
            );
        };

        // ---- Register Account A ("Hybrid-Alpha") & Account B ("Hybrid-Beta") ----
        const browserA = new BrowserGuiClient(apiUrl);
        const browserB = new BrowserGuiClient(apiUrl);
        const alphaEmail = "alpha@solaris.local";
        const betaEmail = "beta@solaris.local";
        const password = "Password123!Solaris";

        await browserA.request("POST", "/api/user/", {
            email: alphaEmail,
            username: "HybridAlphaUser",
            password,
        });
        await browserB.request("POST", "/api/user/", {
            email: betaEmail,
            username: "HybridBetaUser",
            password,
        });
        await container.userService.userRepo.updateMany(
            {},
            { $set: { "achievements.completed": 5 } },
        );

        await browserA.request("POST", "/api/auth/login", {
            email: alphaEmail,
            password,
        });
        await browserB.request("POST", "/api/auth/login", {
            email: betaEmail,
            password,
        });

        // ---- Create Official 10-Player Standard FFA Game (`standard.json`, `standard_rt`) ----
        const standardPath = path.resolve(
            __dirname,
            "../../../config/game/settings/official/standard.json",
        );
        const standardSettings = JSON.parse(
            fs.readFileSync(standardPath, "utf8"),
        );
        standardSettings.general.name = "Admiral Two-Hybrid E2E";
        standardSettings.general.fluxEnabled = "disabled";
        standardSettings.general.timeMachine = "enabled";
        standardSettings.general.joinRandomSlot = "disabled";
        standardSettings.general.playerLimit = 10;
        standardSettings.galaxy.productionTicks = 20;
        standardSettings.galaxy.customSeed = "admiral-two-hybrid-seed-1";
        standardSettings.gameTime.speed = 3600;

        const createdGame = await container.gameCreateService.create(
            container.eventService,
            standardSettings,
            null,
        );
        const gameId = String(createdGame._id);

        // Pick two neighboring slots on the circular 10-player ring and ensure
        // Alpha & Beta share an in-range border pair + nearby unclaimed star.
        const rawGame = await container.gameService.getByIdAll(
            objectIdFromString(gameId),
        );
        assert(rawGame, "Created game must exist in MongoDB");
        const slotAlpha = rawGame.galaxy.players[0];
        const slotBeta = rawGame.galaxy.players[1];
        const slotEnemyAlpha = rawGame.galaxy.players[9];
        const slotEnemyBeta = rawGame.galaxy.players[2];
        const alphaPlayerId = slotAlpha._id.toString();
        const betaPlayerId = slotBeta._id.toString();
        const enemyAlphaId = slotEnemyAlpha._id.toString();
        const enemyBetaId = slotEnemyBeta._id.toString();

        // Join via real HTTP `PUT /api/game/:gameId/join` as Browser A & Browser B.
        const avatarsResp = await browserA.request<any>(
            "GET",
            "/api/user/avatars",
        );
        const avatarList: { id: number; unlocked: boolean }[] = Array.isArray(
            avatarsResp,
        )
            ? avatarsResp
            : (avatarsResp?.avatars ?? []);
        const unlockedAvatarId = avatarList.find((a) => a.unlocked)?.id ?? 12;

        await browserA.request("PUT", `/api/game/${gameId}/join`, {
            playerId: alphaPlayerId,
            alias: "Hybrid-Alpha",
            avatar: unlockedAvatarId,
        });
        await browserB.request("PUT", `/api/game/${gameId}/join`, {
            playerId: betaPlayerId,
            alias: "Hybrid-Beta",
            avatar: unlockedAvatarId,
        });

        // Fill seats 3-10 with built-in Advanced AI & start game immediately.
        const fullGame = await container.gameService.getByIdAll(
            objectIdFromString(gameId),
        );
        assert(fullGame, "Game must load");
        container.gameJoinService.assignNonUserPlayersToAI(fullGame, false);
        container.gameJoinService.startGame(fullGame);
        const startedAt = new Date(Date.now() - 120_000);
        fullGame.state.startDate = startedAt;
        fullGame.state.lastTickDate = startedAt;

        // Place one Alpha star, one Beta star, one unclaimed star, and one Enemy star
        // along the Alpha-Beta / Alpha-Enemy midpoints (~1.2 LY apart) so trade, gifting,
        // covert ToT attacks, and hostile combat are deterministically exercisable.
        const alphaHomeId = fullGame.galaxy.players[0].homeStarId?.toString();
        const betaHomeId = fullGame.galaxy.players[1].homeStarId?.toString();
        const enemyHomeId = fullGame.galaxy.players
            .find((p) => p._id.toString() === enemyAlphaId)
            ?.homeStarId?.toString();
        const alphaStars = fullGame.galaxy.stars.filter(
            (s) =>
                s.ownedByPlayerId?.toString() === alphaPlayerId &&
                s._id.toString() !== alphaHomeId,
        );
        const betaStars = fullGame.galaxy.stars.filter(
            (s) =>
                s.ownedByPlayerId?.toString() === betaPlayerId &&
                s._id.toString() !== betaHomeId,
        );
        const enemyAStars = fullGame.galaxy.stars.filter(
            (s) =>
                s.ownedByPlayerId?.toString() === enemyAlphaId &&
                s._id.toString() !== enemyHomeId,
        );
        const unownedStars = fullGame.galaxy.stars.filter(
            (s) => s.ownedByPlayerId == null,
        );
        const anchorA =
            fullGame.galaxy.stars.find((s) => s._id.toString() === alphaHomeId)
                ?.location ?? alphaStars[0].location;
        const ly = fullGame.constants?.distances?.lightYear || 50;
        const maxGalaxyX = Math.max(
            ...fullGame.galaxy.stars.map((s) => s.location.x),
        );
        // Place the Alpha-Beta diplomatic border pocket 10 LY outside `maxGalaxyX`
        // so random 10-player seat shuffles never drop a third-party AI home cluster
        // on top of `alphaBorderStar` / `betaBorderStar` during Ticks 1..25:
        const midAB = {
            x: maxGalaxyX + 10 * ly,
            y: 0,
        };
        const alphaBorderStar = alphaStars[0];
        const alphaSecondStar = alphaStars[1];
        const betaBorderStar = betaStars[0];
        const enemyBorderStar = enemyAStars[0];
        const sharedUnownedStar = unownedStars[0];
        const ignoredStar = alphaStars[2];

        alphaBorderStar.location = { x: midAB.x - 0.6 * ly, y: midAB.y };
        alphaBorderStar.warpGate = false;
        alphaBorderStar.wormHoleToStarId = null;
        alphaSecondStar.location = {
            x: midAB.x - 1.0 * ly,
            y: midAB.y + 0.6 * ly,
        };
        alphaSecondStar.warpGate = false;
        alphaSecondStar.wormHoleToStarId = null;
        betaBorderStar.location = { x: midAB.x + 0.6 * ly, y: midAB.y };
        betaBorderStar.warpGate = false;
        betaBorderStar.wormHoleToStarId = null;
        sharedUnownedStar.location = {
            x: midAB.x,
            y: midAB.y + 0.8 * ly,
        };
        sharedUnownedStar.warpGate = false;
        sharedUnownedStar.wormHoleToStarId = null;
        enemyBorderStar.location = {
            x: anchorA.x - 1.2 * ly,
            y: anchorA.y - 0.4 * ly,
        };
        ignoredStar.shipsActual = 150;
        ignoredStar.ships = 150;
        for (const c of fullGame.galaxy.carriers) {
            if (c.orbiting) {
                const st = fullGame.galaxy.stars.find(
                    (s) => s._id.toString() === c.orbiting!.toString(),
                );
                if (st) c.location = { ...st.location };
            }
        }
        await fullGame.save();
        await container.historyService.log(fullGame);

        const alphaBorderStarId = alphaBorderStar._id.toString();
        const betaBorderStarId = betaBorderStar._id.toString();
        const ignoredStarId = ignoredStar._id.toString();
        const ignoredStarInitialInfra = {
            economy: ignoredStar.infrastructure.economy ?? 0,
            industry: ignoredStar.infrastructure.industry ?? 0,
            science: ignoredStar.infrastructure.science ?? 0,
        };

        // ---- Boot Daemon Alpha & Daemon Beta (first in `advisor` dry-run mode) ----
        const alphaAlerts: LiveAlert[] = [];
        const betaAlerts: LiveAlert[] = [];

        const daemonAlpha = registerDaemon(
            new LiveAdmiralDaemon({
                apiUrl,
                gameId,
                email: alphaEmail,
                password,
                checkpointPath: ckptPath,
                statePath: alphaDbPath,
                mode: "advisor",
                cashReserveCredits: 150,
                neutralBorderPosture: "defend",
                maxRequestsPerSecond: 2,
                tickOffsetMs: 500,
                pollIntervalMs: 1000,
                deadlineSafetyMs: 2000,
                carrierGraceTicks: 2,
                allowEmergencyDefenseReserveDip: true,
                onAlert: (a) => alphaAlerts.push(a),
            }),
        );

        const daemonBeta = registerDaemon(
            new LiveAdmiralDaemon({
                apiUrl,
                gameId,
                email: betaEmail,
                password,
                checkpointPath: ckptPath,
                statePath: betaDbPath,
                mode: "advisor",
                cashReserveCredits: 150,
                neutralBorderPosture: "defend",
                maxRequestsPerSecond: 2,
                tickOffsetMs: 500,
                pollIntervalMs: 1000,
                deadlineSafetyMs: 2000,
                carrierGraceTicks: 2,
                allowEmergencyDefenseReserveDip: true,
                onAlert: (a) => betaAlerts.push(a),
            }),
        );

        await daemonAlpha.initialize();
        await daemonBeta.initialize();

        // Verify Multi-Client Session Isolation: Browser A and Daemon Alpha stay logged in simultaneously!
        assert.notEqual(
            browserA.getCookie(),
            daemonAlpha.client.getCookie(),
            "Daemon login must use an independent session cookie from Browser A",
        );
        const [verifyBrowserA, verifyDaemonA] = await Promise.all([
            browserA.request<{ _id: string }>("POST", "/api/auth/verify", {}),
            daemonAlpha.client.verifySession(),
        ]);
        assert.equal(verifyBrowserA._id, verifyDaemonA._id);

        // =====================================================================
        // SCENARIO 1: Initial Neutral Peace (`advisor` dry-run -> `approval` -> `autonomous`)
        // =====================================================================
        console.log(
            "[E2E] 2/10 Scenario 1: Initial Neutral Peace (advisor -> approval -> autonomous)...",
        );
        await advanceServerTick(); // Tick 1 (cycle start)
        const preAdvisorGal = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const preCreditsA = preAdvisorGal.galaxy.players[0].credits;

        const advisorRepA = await daemonAlpha.runOnce("advisor_dry_run");
        const advisorRepB = await daemonBeta.runOnce("advisor_dry_run");
        assert.equal(advisorRepA.mode, "advisor");
        assert.equal(advisorRepA.dispatchedCount, 0);
        assert.equal(advisorRepB.dispatchedCount, 0);
        assert(
            advisorRepA.plannedActions.length > 0,
            "Advisor should plan real Tick-1 actions",
        );
        const postAdvisorGal = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        assert.equal(
            postAdvisorGal.galaxy.players[0].credits,
            preCreditsA,
            "Advisor dry-run must perform ZERO live mutations",
        );

        // Test `approval` mode gate (rejecting vs approving):
        daemonAlpha.config.mode = "approval";
        daemonAlpha.config.approvalHandler = async () => false;
        const rejectedRep = await daemonAlpha.runOnce("approval_reject");
        assert.equal(rejectedRep.dispatchedCount, 0);

        // Declare 1 AI neighbor as `enemy` for Alpha and 1 for Beta in Browser GUI;
        // toggle `ignoreBulkUpgrade` on `ignoredStarId`; switch Notes to `autonomous` + `cash_reserve: 150`.
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/diplomacy/enemy/${enemyAlphaId}`,
            {},
        );
        await browserB.request(
            "PUT",
            `/api/game/${gameId}/diplomacy/enemy/${enemyBetaId}`,
            {},
        );
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/star/toggleignorebulkupgradeall`,
            { starId: ignoredStarId, ignoreStatus: true },
        );
        const baseNotes = [
            "Personal human strategy scratchpad (ignored by parser).",
            "# === AI ===",
            "mode: autonomous",
            "cash_reserve: 150",
            "neutral_border: defend",
            "# === END AI ===",
        ].join("\n");
        await browserA.request("PUT", `/api/game/${gameId}/notes`, {
            notes: baseNotes,
        });
        await browserB.request("PUT", `/api/game/${gameId}/notes`, {
            notes: baseNotes,
        });

        const autoRepA = await daemonAlpha.runOnce("s1_auto");
        const autoRepB = await daemonBeta.runOnce("s1_auto");
        assert.equal(autoRepA.mode, "autonomous");
        assert.equal(autoRepB.mode, "autonomous");
        assert(
            autoRepA.dispatchedCount > 0,
            "Autonomous Tick 1 must dispatch actions",
        );
        assert.equal(
            autoRepA.diplomacy.entriesByPlayerId.get(enemyAlphaId)?.tier,
            "HOSTILE",
        );
        assert.equal(
            autoRepA.diplomacy.entriesByPlayerId.get(betaPlayerId)?.tier,
            "PROTECTED",
        );

        const s1GalA = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        assert(
            Array.isArray(s1GalA.galaxy.players[0].researchQueue) &&
                s1GalA.galaxy.players[0].researchQueue.length > 0 &&
                !s1GalA.galaxy.players[0].researchQueue.includes("random"),
            `Expected non-random researchQueue on live server after update/258 sync, got ${JSON.stringify(s1GalA.galaxy.players[0].researchQueue)}`,
        );
        const starOwnerMap = new Map<string, string | null>();
        for (const s of s1GalA.galaxy.stars) {
            starOwnerMap.set(
                String(s._id),
                s.ownedByPlayerId ? String(s.ownedByPlayerId) : null,
            );
        }
        for (const c of s1GalA.galaxy.carriers.filter(
            (x: any) => String(x.ownedByPlayerId) === alphaPlayerId,
        )) {
            for (const wp of c.waypoints ?? []) {
                const dstOwner = starOwnerMap.get(String(wp.destination));
                if (dstOwner && dstOwner !== alphaPlayerId) {
                    assert.equal(
                        dstOwner,
                        enemyAlphaId,
                        `Alpha carrier must NEVER target neutral/protected player ${dstOwner}`,
                    );
                }
            }
        }
        scenariosVerified.push(
            "Scenario 1: Initial Neutral Peace (advisor -> approval -> autonomous)",
        );

        // =====================================================================
        // SCENARIO 2: Pending Alliance Offer -> Formal Alliance
        // =====================================================================
        console.log(
            "[E2E] 3/10 Scenario 2: Pending Alliance Offer -> Formal Alliance...",
        );
        // Alpha clicks Ally on Beta; Beta has NOT accepted yet.
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/diplomacy/ally/${betaPlayerId}`,
            {},
        );
        const pendingRepA = await daemonAlpha.runOnce("s2_pending_ally");
        const betaDipPending =
            pendingRepA.diplomacy.entriesByPlayerId.get(betaPlayerId)!;
        assert.equal(betaDipPending.statusTo, "allies");
        assert.equal(betaDipPending.actualStatus, "neutral");
        assert.equal(
            betaDipPending.tier,
            "PROTECTED",
            "Pending unilateral alliance offer must stay PROTECTED",
        );

        // Beta accepts Alliance in GUI -> both become `ALLIED`.
        await browserB.request(
            "PUT",
            `/api/game/${gameId}/diplomacy/ally/${alphaPlayerId}`,
            {},
        );
        const alliedRepA = await daemonAlpha.runOnce("s2_allied");
        const alliedRepB = await daemonBeta.runOnce("s2_allied");
        assert.equal(
            alliedRepA.diplomacy.entriesByPlayerId.get(betaPlayerId)?.tier,
            "ALLIED",
        );
        assert.equal(
            alliedRepB.diplomacy.entriesByPlayerId.get(alphaPlayerId)?.tier,
            "ALLIED",
        );
        scenariosVerified.push(
            "Scenario 2: Pending Alliance Offer -> Formal Alliance",
        );

        // =====================================================================
        // SCENARIO 3: Cash Reserve (`cash_reserve: 150`) Across Tick 1 & Tick 19 + Manual Trade
        // =====================================================================
        console.log(
            "[E2E] 4/10 Scenario 3: Cash Reserve across Cycle 1 (Tick 1 -> Tick 19) & Manual Credit Trade...",
        );
        const postTick1A = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const postTick1B = await browserB.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        assert(
            postTick1A.galaxy.players[0].credits >= 150,
            `Alpha Tick-1 credits (${postTick1A.galaxy.players[0].credits}) must respect cash_reserve: 150`,
        );
        assert(
            postTick1B.galaxy.players[1].credits >= 150,
            `Beta Tick-1 credits (${postTick1B.galaxy.players[1].credits}) must respect cash_reserve: 150`,
        );

        // Advance server ticks up to Tick 19 (Economy upgrade tick in 20-tick cycle):
        while (true) {
            const st = await browserA.request(
                "GET",
                `/api/game/${gameId}/state`,
            );
            if (st.state.tick >= 19) break;
            await advanceServerTick();
        }
        await daemonAlpha.runOnce("s3_tick19");
        await daemonBeta.runOnce("s3_tick19");

        const postTick19A = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const postTick19B = await browserB.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        assert(
            postTick19A.galaxy.players[0].credits >= 150,
            `Alpha Tick-19 credits (${postTick19A.galaxy.players[0].credits}) must stay >= 150`,
        );
        assert(
            postTick19B.galaxy.players[1].credits >= 150,
            `Beta Tick-19 credits (${postTick19B.galaxy.players[1].credits}) must stay >= 150`,
        );

        // Verify `star.ignoreBulkUpgrade` was respected across both Tick 1 and Tick 19!
        const ignoredStarAfter = postTick19A.galaxy.stars.find(
            (s: any) => String(s._id) === ignoredStarId,
        );
        assert.deepEqual(
            ignoredStarAfter.infrastructure,
            ignoredStarInitialInfra,
            "Star with ignoreBulkUpgrade enabled must not be upgraded by Admiral",
        );

        // Alpha immediately executes a manual GUI credit send out of the protected reserve:
        await browserA.request("PUT", `/api/game/${gameId}/trade/credits`, {
            toPlayerId: betaPlayerId,
            amount: 50,
        });
        await waitUntil(
            async () => betaAlerts.some((a) => a.code === "CREDITS_RECEIVED"),
            4000,
        );
        scenariosVerified.push(
            "Scenario 3: Cash Reserve (Tick 1 & Tick 19 >= 150cr), ignoreBulkUpgrade & Manual Credit Trade",
        );

        // =====================================================================
        // SCENARIO 4: Mid-Cycle Tech Reception via Real-Time Socket.IO (`playerTechnologyReceived`)
        // =====================================================================
        console.log(
            "[E2E] 5/10 Scenario 4: Mid-Cycle Tech Reception via Socket.IO...",
        );
        // Give Alpha level 3 `weapons` and level 3 `terraforming` (while Beta has level 1 in both)
        // and set Beta's `researchingNow = "weapons"` (10% progress) synced with `daemonBeta`.
        {
            const dbGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(dbGame);
            const pA = dbGame.galaxy.players[0];
            const pB = dbGame.galaxy.players[1];
            pA.credits = Math.max(pA.credits, 300);
            pA.research.weapons.level = 3;
            pA.research.terraforming.level = 3;
            pB.research.weapons.level = 1;
            pB.research.weapons.progress = 12;
            pB.research.terraforming.level = 1;
            pB.researchingNow = "weapons";
            await dbGame.save();
        }
        // Sync Beta's baseline so `weapons` is tracked as AI-known (not a manual pin)
        daemonBeta.getPersistedState().human.lastKnownTechLevels["weapons"] = 1;
        daemonBeta.getPersistedState().human.lastAiResearch = {
            now: "weapons",
            next: "banking",
            tick: 19,
        };
        daemonBeta.getPersistedState().human.humanResearchPin = null;

        // Browser A sends `weapons` level 2 to Beta via real HTTP `PUT /api/game/:gameId/trade/tech`.
        // Do NOT call `daemonBeta.runOnce()` — let Socket.IO `playerTechnologyReceived` fire automatically!
        await browserA.request("PUT", `/api/game/${gameId}/trade/tech`, {
            toPlayerId: betaPlayerId,
            technology: "weapons",
            level: 2,
        });

        await waitUntil(async () => {
            const gB = await browserB.request(
                "GET",
                `/api/game/${gameId}/galaxy`,
            );
            return gB.galaxy.players[1].researchingNow !== "weapons";
        }, 6000);

        const postTechGalB = await browserB.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const newBetaNow = postTechGalB.galaxy.players[1].researchingNow;
        assert.notEqual(
            newBetaNow,
            "weapons",
            "Beta's daemon must switch off wiped `weapons` within seconds via Socket.IO",
        );
        assert.notEqual(
            newBetaNow,
            "terraforming",
            "Beta's daemon must avoid `terraforming` because in-range ally Alpha already holds level 3",
        );
        assert(
            betaAlerts.some((a) => a.code === "TRADE_OPPORTUNITY"),
            "Beta's daemon should emit TRADE_OPPORTUNITY for ally's higher tech",
        );
        scenariosVerified.push(
            "Scenario 4: Real-Time Socket.IO Mid-Cycle Tech Reception & Ally Tech-Gap Filter",
        );

        // =====================================================================
        // SCENARIO 5: Manual GUI Research Pin
        // =====================================================================
        console.log("[E2E] 6/10 Scenario 5: Manual GUI Research Pin...");
        const manualPinTech =
            newBetaNow === "hyperspace" ? "scanning" : "hyperspace";
        await browserB.request("PUT", `/api/game/${gameId}/research/now`, {
            preference: manualPinTech,
        });
        await daemonBeta.runOnce("s5_detect_pin");
        assert.equal(
            daemonBeta.getPersistedState().human.humanResearchPin?.tech,
            manualPinTech,
            "Daemon Beta must detect human GUI research change and pin it",
        );

        // Advance across cycle boundary (Tick 20 -> Tick 21) and verify pin holds!
        await advanceServerTick(); // Tick 20
        await advanceServerTick(); // Tick 21
        await daemonBeta.runOnce("s5_cycle_boundary");
        const postPinGalB = await browserB.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        assert.equal(
            postPinGalB.galaxy.players[1].researchingNow,
            manualPinTech,
            "Manual GUI research pin must be preserved across cycle boundary",
        );
        scenariosVerified.push(
            "Scenario 5: Manual GUI Research Pin Across Cycle Boundary",
        );

        // =====================================================================
        // SCENARIO 6: Carrier Staging & Gifting (`humanPinnedCarriers` + `isGift: true`)
        // =====================================================================
        console.log("[E2E] 7/10 Scenario 6: Carrier Staging & Gifting...");
        // Ensure `alphaBorderStar` has >= 45 ships and `betaBorderStar` has 10 ships (with no carriers orbiting alphaBorderStar).
        {
            const dbGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(dbGame);
            dbGame.galaxy.players[0].credits = Math.max(
                dbGame.galaxy.players[0].credits,
                300,
            );
            const aStar = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === alphaBorderStarId,
            )!;
            aStar.ownedByPlayerId = slotAlpha._id;
            aStar.warpGate = false;
            aStar.shipsActual = 50;
            aStar.ships = 50;
            const bStar = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === betaBorderStarId,
            )!;
            bStar.ownedByPlayerId = slotBeta._id;
            bStar.warpGate = false;
            bStar.shipsActual = 10;
            bStar.ships = 10;
            await dbGame.save();
        }

        // Alpha human builds a carrier in Browser GUI with 30 ships and waits 1 tick WITHOUT setting waypoints:
        const builtGiftCar = await browserA.request<{
            carrier: { _id: string; ships: number };
        }>("PUT", `/api/game/${gameId}/star/build/carrier`, {
            starId: alphaBorderStarId,
            ships: 30,
        });
        const stagedCarrierId = String(builtGiftCar.carrier._id);

        await advanceServerTick(); // Tick 22
        await daemonAlpha.runOnce("s6_stage_wait");

        const postStageGalA = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const stagedCarAfter = postStageGalA.galaxy.carriers.find(
            (c: any) => String(c._id) === stagedCarrierId,
        );
        assert(stagedCarAfter, "Human-staged carrier must still exist");
        assert(
            stagedCarAfter.ships >= 30,
            `Human-staged carrier must NEVER have its 30 ships stripped (got ${stagedCarAfter.ships})`,
        );
        assert(
            daemonAlpha.getPersistedState().human.humanPinnedCarriers[
                stagedCarrierId
            ],
            "Staged carrier must be tracked in humanPinnedCarriers",
        );

        // Now Alpha plots route to Beta's star and clicks Gift Carrier in GUI!
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/carrier/${stagedCarrierId}/waypoints`,
            {
                waypoints: [
                    {
                        source: alphaBorderStarId,
                        destination: betaBorderStarId,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                looped: false,
            },
        );
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/carrier/${stagedCarrierId}/gift`,
            {},
        );

        // Advance 1 tick so the 30-ship `isGift: true` carrier launches into space toward Beta's 10-ship star!
        await advanceServerTick(); // Tick 23 (ETA <= 3 ticks -> inside doomed-star evacuation window if treated as hostile!)
        const surpriseBeforeCount = betaAlerts.filter(
            (a) => a.code === "SURPRISE_ATTACK",
        ).length;
        const giftRepB = await daemonBeta.runOnce("s6_inbound_gift");
        const surpriseAfterCount = betaAlerts.filter(
            (a) => a.code === "SURPRISE_ATTACK",
        ).length;
        assert.equal(
            surpriseAfterCount,
            surpriseBeforeCount,
            "Inbound `isGift: true` carrier must NEVER trigger SURPRISE_ATTACK",
        );
        const panicActions = giftRepB.plannedActions.filter(
            (a) =>
                a.type === "SaveWaypoints" &&
                a.isDefense === true &&
                a.waypoints.some(
                    (w) =>
                        w.source === betaBorderStarId ||
                        w.destination === betaBorderStarId,
                ),
        );
        assert.equal(
            panicActions.length,
            0,
            "Beta must NOT trigger defense reinforcements or doomed-star evacuation against inbound gift carrier",
        );
        scenariosVerified.push(
            "Scenario 6: Human Carrier Staging Grace Pin & Inbound Gift Carrier Non-Panic",
        );

        // =====================================================================
        // SCENARIO 7: Formal-Ally Hard-Block, Covert Attack (`attack: [Hybrid-Beta]`),
        //             Peace Mid-Attack Cancellation + Alert, & Surprise Neutral Betrayal
        // =====================================================================
        console.log(
            "[E2E] 8/10 Scenario 7: Covert Attack, Ally Hard-Block, Peace Mid-Attack & Neutral Betrayal...",
        );
        // 7a-i: While Alpha & Beta are still formal `allies`, Alpha accidentally lists `Hybrid-Beta` in `attack:`!
        await browserA.request("PUT", `/api/game/${gameId}/notes`, {
            notes: [
                "# === AI ===",
                "mode: autonomous",
                "cash_reserve: 150",
                "attack: [Hybrid-Beta]",
                "# === END AI ===",
            ].join("\n"),
        });
        const allyBlockRep = await daemonAlpha.runOnce("s7_ally_hard_block");
        assert.equal(
            allyBlockRep.diplomacy.entriesByPlayerId.get(betaPlayerId)?.tier,
            "ALLIED",
            "Formal `allies` must NEVER become HOSTILE even when listed in Notes `attack:`",
        );
        assert(
            alphaAlerts.some((a) => a.code === "ALLY_IN_ATTACK_IGNORED"),
            "Listing a formal ally in `attack:` must emit ALLY_IN_ATTACK_IGNORED",
        );

        // 7a-ii: Alpha switches diplomatic status with Beta to `neutral` (NOT `enemy`!),
        // keeping `attack: [Hybrid-Beta]` in private Notes -> Covert Attack!
        await browserA.request(
            "PUT",
            `/api/game/${gameId}/diplomacy/neutral/${betaPlayerId}`,
            {},
        );
        {
            // Give Alpha strong ships on its two nearby stars (`alphaBorderStar` and `alphaSecondStar`)
            // and place Beta's star well within jump range with light garrison so Admiral launches a covert attack!
            const dbGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(dbGame);
            dbGame.galaxy.players[0].credits = Math.max(
                dbGame.galaxy.players[0].credits,
                350,
            );
            const a1 = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === alphaBorderStarId,
            )!;
            const a2 = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === alphaSecondStar._id.toString(),
            )!;
            const b1 = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === betaBorderStarId,
            )!;
            a1.shipsActual = 90;
            a1.ships = 90;
            a2.shipsActual = 90;
            a2.ships = 90;
            b1.shipsActual = 5;
            b1.ships = 5;
            await dbGame.save();
        }

        const covertRep = await daemonAlpha.runOnce("s7_covert_attack");
        assert.equal(
            covertRep.diplomacy.entriesByPlayerId.get(betaPlayerId)?.tier,
            "HOSTILE",
        );
        assert.equal(
            covertRep.diplomacy.entriesByPlayerId.get(betaPlayerId)
                ?.covertAttack,
            true,
        );
        // Verify Beta's diplomacy view STILL shows `actualStatus === "neutral"` (zero war declaration leaked!):
        const betaDipList = await browserB.request<any[]>(
            "GET",
            `/api/game/${gameId}/diplomacy`,
        );
        const alphaStatusFromBeta = betaDipList.find(
            (d) => String(d.playerIdTo ?? d.playerId) === alphaPlayerId,
        );
        assert.equal(
            alphaStatusFromBeta?.actualStatus,
            "neutral",
            "Covert attack must NEVER call declareEnemy or alter formal neutral status",
        );

        // Ensure we have BOTH an in-flight AI carrier headed to `betaBorderStarId` AND
        // an orbiting AI-committed carrier with `delayTicks > 0` headed to `betaBorderStarId`
        // so removing `Hybrid-Beta` from `attack:` exercises BOTH branches of Invariant 2!
        {
            const dbGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(dbGame);
            let alphaOwnedCars = dbGame.galaxy.carriers.filter(
                (c) =>
                    c.ownedByPlayerId?.toString() === alphaPlayerId &&
                    !c.isGift &&
                    !daemonAlpha.getPersistedState().human.humanPinnedCarriers[
                        c._id.toString()
                    ],
            );
            // Guarantee at least 2 AI-controlled carriers on Alpha to test both in-flight + delayed orbital cancellation:
            while (alphaOwnedCars.length < 2) {
                await daemonAlpha.client.buildCarrier(
                    gameId,
                    alphaBorderStarId,
                    20,
                    `s7-setup-car-${alphaOwnedCars.length}-${Date.now()}`,
                );
                const reloaded = await container.gameService.getByIdAll(
                    objectIdFromString(gameId),
                );
                assert(reloaded);
                alphaOwnedCars = reloaded.galaxy.carriers.filter(
                    (c) =>
                        c.ownedByPlayerId?.toString() === alphaPlayerId &&
                        !c.isGift &&
                        !daemonAlpha.getPersistedState().human
                            .humanPinnedCarriers[c._id.toString()],
                );
            }
            const freshGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(freshGame);
            const aiCars = freshGame.galaxy.carriers.filter(
                (c) =>
                    c.ownedByPlayerId?.toString() === alphaPlayerId &&
                    !c.isGift &&
                    !daemonAlpha.getPersistedState().human.humanPinnedCarriers[
                        c._id.toString()
                    ],
            );
            const inFlightCar = aiCars[0];
            const delayedOrbitCar = aiCars[1];

            // Launch `inFlightCar` mid-space toward `betaBorderStar` (with a follow-on waypoint too)
            inFlightCar.orbiting = null;
            inFlightCar.location = {
                x: (alphaBorderStar.location.x + betaBorderStar.location.x) / 2,
                y: alphaBorderStar.location.y,
            };
            inFlightCar.waypoints = [
                {
                    _id: delayedOrbitCar._id,
                    source: alphaBorderStar._id,
                    destination: betaBorderStar._id,
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 0,
                } as any,
                {
                    _id: inFlightCar._id,
                    source: betaBorderStar._id,
                    destination: alphaSecondStar._id,
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 0,
                } as any,
            ];

            // Station `delayedOrbitCar` in orbit at `alphaSecondStar` with `delayTicks: 2` ToT waypoint -> `betaBorderStar`
            delayedOrbitCar.orbiting = alphaSecondStar._id;
            delayedOrbitCar.location = { ...alphaSecondStar.location };
            delayedOrbitCar.waypoints = [
                {
                    _id: alphaSecondStar._id,
                    source: alphaSecondStar._id,
                    destination: betaBorderStar._id,
                    action: "nothing",
                    actionShips: 0,
                    delayTicks: 2,
                } as any,
            ];
            await freshGame.save();

            // Record both in `aiCommittedCarriers` as AI-owned covert-attack orders:
            const curTick = freshGame.state.tick;
            daemonAlpha.getPersistedState().human.aiCommittedCarriers[
                inFlightCar._id.toString()
            ] = snapshotCarrier(inFlightCar, curTick);
            daemonAlpha.getPersistedState().human.aiCommittedCarriers[
                delayedOrbitCar._id.toString()
            ] = snapshotCarrier(delayedOrbitCar, curTick);

            // 7a-iii: Peace Mid-Attack — Alpha removes `Hybrid-Beta` from `attack:` in Notes!
            await browserA.request("PUT", `/api/game/${gameId}/notes`, {
                notes: baseNotes,
            });
            await daemonAlpha.runOnce("s7_peace_mid_attack");

            const postPeaceGalA = await browserA.request(
                "GET",
                `/api/game/${gameId}/galaxy`,
            );
            const delayedAfter = postPeaceGalA.galaxy.carriers.find(
                (c: any) => String(c._id) === delayedOrbitCar._id.toString(),
            );
            const inFlightAfter = postPeaceGalA.galaxy.carriers.find(
                (c: any) => String(c._id) === inFlightCar._id.toString(),
            );
            assert.equal(
                delayedAfter.waypoints.length,
                0,
                "Orbiting delayed ToT carrier targeting newly PROTECTED player must have waypoints cleared to []",
            );
            assert.equal(
                inFlightAfter.waypoints.length,
                1,
                "In-flight carrier must have follow-on waypoints[1..] truncated while keeping locked hop[0]",
            );
            assert(
                alphaAlerts.some((a) => a.code === "UNRECALLABLE_FLEET"),
                "In-flight carrier headed to newly PROTECTED star must emit urgent UNRECALLABLE_FLEET alert",
            );
        }

        // 7b: Surprise Neutral Betrayal — Beta (already `neutral`!) launches a non-gift carrier at Alpha's star!
        {
            const dbGame = await container.gameService.getByIdAll(
                objectIdFromString(gameId),
            );
            assert(dbGame);
            dbGame.galaxy.players[1].credits = Math.max(
                dbGame.galaxy.players[1].credits,
                250,
            );
            const aStar = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === alphaBorderStarId,
            )!;
            const bStar = dbGame.galaxy.stars.find(
                (s) => s._id.toString() === betaBorderStarId,
            )!;
            aStar.ownedByPlayerId = slotAlpha._id;
            aStar.warpGate = false;
            aStar.naturalResources = { economy: 30, industry: 30, science: 30 };
            aStar.shipsActual = 120;
            aStar.ships = 120;
            bStar.ownedByPlayerId = slotBeta._id;
            bStar.warpGate = false;
            bStar.naturalResources = { economy: 30, industry: 30, science: 30 };
            bStar.shipsActual = 60;
            bStar.ships = 60;
            const alphaOwnedStarIds = new Set(
                dbGame.galaxy.stars
                    .filter(
                        (s) => s.ownedByPlayerId?.toString() === alphaPlayerId,
                    )
                    .map((s) => s._id.toString()),
            );
            const betaOwnedStarIds = new Set(
                dbGame.galaxy.stars
                    .filter(
                        (s) => s.ownedByPlayerId?.toString() === betaPlayerId,
                    )
                    .map((s) => s._id.toString()),
            );
            // Park any leftover in-flight or former-ally co-orbiting Alpha/Beta carriers
            // (allies can co-orbit each other's stars during Scenarios 2-6, which triggers
            // star combat + `reputation.ts:167` auto-war once neutral ticks advance!):
            for (const c of dbGame.galaxy.carriers) {
                const owner = c.ownedByPlayerId?.toString();
                const orb = c.orbiting?.toString();
                const dst = c.waypoints[0]?.destination?.toString();
                if (owner === alphaPlayerId) {
                    if (
                        !orb ||
                        !alphaOwnedStarIds.has(orb) ||
                        c.waypoints.length > 0
                    ) {
                        c.orbiting = aStar._id;
                        c.location = { ...aStar.location };
                        c.waypoints = [];
                    }
                } else if (owner === betaPlayerId) {
                    if (
                        !orb ||
                        !betaOwnedStarIds.has(orb) ||
                        c.waypoints.length > 0
                    ) {
                        c.orbiting = bStar._id;
                        c.location = { ...bStar.location };
                        c.waypoints = [];
                    }
                } else if (
                    dst === alphaBorderStarId ||
                    dst === betaBorderStarId
                ) {
                    const homeStar = dbGame.galaxy.stars.find(
                        (s) => s.ownedByPlayerId?.toString() === owner,
                    );
                    if (homeStar) {
                        c.orbiting = homeStar._id;
                        c.location = { ...homeStar.location };
                        c.waypoints = [];
                    }
                }
            }
            await dbGame.save();
        }
        const betrayalBuild = await browserB.request<{
            carrier: { _id: string };
        }>("PUT", `/api/game/${gameId}/star/build/carrier`, {
            starId: betaBorderStarId,
            ships: 40,
        });
        await browserB.request(
            "PUT",
            `/api/game/${gameId}/carrier/${betrayalBuild.carrier._id}/waypoints`,
            {
                waypoints: [
                    {
                        source: betaBorderStarId,
                        destination: alphaBorderStarId,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 0,
                    },
                ],
                looped: false,
            },
        );
        const betaSurpriseBefore = alphaAlerts.filter(
            (a) =>
                a.code === "SURPRISE_ATTACK" &&
                a.details?.attackerPlayerId === betaPlayerId,
        ).length;
        await advanceServerTick(); // Launch Beta's neutral attack carrier into transit (ETA 3 ticks)!
        const diagGalA = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        const betrayalRepA = await daemonAlpha.runOnce("s7_betrayal_defense");
        const betaSurpriseAfter = alphaAlerts.filter(
            (a) =>
                a.code === "SURPRISE_ATTACK" &&
                a.details?.attackerPlayerId === betaPlayerId,
        ).length;
        assert(
            betaSurpriseAfter > betaSurpriseBefore,
            `Neutral non-gift attack carrier from Beta must trigger SURPRISE_ATTACK alert; diag=${JSON.stringify(
                {
                    dipEntry:
                        betrayalRepA.diplomacy.entriesByPlayerId.get(
                            betaPlayerId,
                        ),
                    aStarOwner: diagGalA.galaxy.stars.find(
                        (s: any) => String(s._id) === alphaBorderStarId,
                    )?.ownedByPlayerId,
                    betaCarriersSeenByAlpha: diagGalA.galaxy.carriers.filter(
                        (c: any) => String(c.ownedByPlayerId) === betaPlayerId,
                    ),
                },
            )}`,
        );
        assert.equal(
            betrayalRepA.diplomacy.entriesByPlayerId.get(betaPlayerId)?.tier,
            "PROTECTED",
            "Surprise neutral betrayal must defend locally WITHOUT auto-declaring war",
        );
        const postBetrayalGalA = await browserA.request(
            "GET",
            `/api/game/${gameId}/galaxy`,
        );
        for (const c of postBetrayalGalA.galaxy.carriers.filter(
            (x: any) =>
                String(x.ownedByPlayerId) === alphaPlayerId &&
                x.orbiting != null,
        )) {
            for (const wp of c.waypoints ?? []) {
                assert.notEqual(
                    String(wp.destination),
                    betaBorderStarId,
                    "Alpha must NOT counter-invade Beta's stars while Beta remains PROTECTED",
                );
            }
        }
        scenariosVerified.push(
            "Scenario 7: Ally Hard-Block, Covert Attack, Peace Mid-Attack Recall/Alert & Surprise Betrayal Defense",
        );

        // =====================================================================
        // SCENARIO 8: Concurrent GUI `400` Conflict Recovery, Crash/Restart SQLite
        //             Persistence + Idempotency Dedup, & Time-Machine Backfill
        // =====================================================================
        console.log(
            "[E2E] 9/10 Scenario 8: Concurrent GUI 400 Recovery, SQLite Crash Restart & Time-Machine Backfill...",
        );
        // 8a: Trigger a real concurrent human GUI mutation mid-planning so the live Solaris server
        // returns HTTP `400 ValidationError` on Daemon Alpha's stale action, verifying clean abort + 1x replan!
        {
            // Advance to Cycle 2 Tick 1 (`tick % 20 === 1` -> Tick 41 or give credits & force Tick 1 of next cycle)
            // Even simpler: advance to next cycle start (Tick 41) OR trigger on a carrier/star action!
            // Let's advance to Tick 41? No, 17 ticks with 8 AI takes ~1.5s total! Let's check or trigger on carrier transfer/upgrade:
            // Actually, if Alpha has an AI-controlled carrier in orbit with 25 ships and Browser A scuttles that carrier
            // right after `daemonAlpha.client.getGalaxy` returns on the first pass, `SaveWaypoints` / `TransferShips`
            // hits a 100% genuine HTTP 400 from Solaris!
            const galNow = await browserA.request(
                "GET",
                `/api/game/${gameId}/galaxy`,
            );
            let targetCar = galNow.galaxy.carriers.find(
                (c: any) =>
                    String(c.ownedByPlayerId) === alphaPlayerId &&
                    c.orbiting != null &&
                    !c.isGift &&
                    !daemonAlpha.getPersistedState().human.humanPinnedCarriers[
                        String(c._id)
                    ],
            );
            if (!targetCar) {
                const b = await daemonAlpha.client.buildCarrier(
                    gameId,
                    alphaBorderStarId,
                    15,
                    `s8-car-${Date.now()}`,
                );
                const g2 = await browserA.request(
                    "GET",
                    `/api/game/${gameId}/galaxy`,
                );
                targetCar = g2.galaxy.carriers.find(
                    (c: any) => String(c._id) === String(b.carrier._id),
                );
                daemonAlpha.getPersistedState().human.aiCommittedCarriers[
                    String(targetCar._id)
                ] = snapshotCarrier(targetCar, g2.state.tick);
            }

            // Give `targetCar` an AI-committed waypoint to `betaBorderStarId` (which is PROTECTED!)
            // so `daemonAlpha` is guaranteed to emit `SaveWaypoints(targetCar._id, [])`, and have
            // `Browser A` concurrently scuttle `targetCar` right after `getGalaxy` resolves!
            await daemonAlpha.client.saveWaypoints(
                gameId,
                String(targetCar._id),
                [
                    {
                        source: String(targetCar.orbiting),
                        destination: betaBorderStarId,
                        action: "nothing",
                        actionShips: 0,
                        delayTicks: 2,
                    },
                ],
                false,
                `s8-stage-wp-${Date.now()}`,
            );
            const gWithWp = await browserA.request(
                "GET",
                `/api/game/${gameId}/galaxy`,
            );
            const refreshedCar = gWithWp.galaxy.carriers.find(
                (c: any) => String(c._id) === String(targetCar._id),
            );
            daemonAlpha.getPersistedState().human.aiCommittedCarriers[
                String(targetCar._id)
            ] = snapshotCarrier(refreshedCar, gWithWp.state.tick);

            const origGetGalaxy = daemonAlpha.client.getGalaxy.bind(
                daemonAlpha.client,
            );
            let injectedConcurrentScuttle = false;
            daemonAlpha.client.getGalaxy = async (gid: string, t?: number) => {
                const res = await origGetGalaxy(gid, t);
                if (!injectedConcurrentScuttle && t == null) {
                    injectedConcurrentScuttle = true;
                    // Human clicks Scuttle Carrier in Browser GUI AFTER daemon fetched `/galaxy`!
                    await browserA.request(
                        "DELETE",
                        `/api/game/${gameId}/carrier/${targetCar._id}/scuttle`,
                    );
                }
                return res;
            };

            try {
                const conflictRep =
                    await daemonAlpha.runOnce("s8_concurrent_gui");
                assert.equal(
                    conflictRep.replanTriggered,
                    true,
                    "Concurrent human GUI 400 must trigger clean abort + single re-plan",
                );
                assert.equal(
                    conflictRep.abortedOnConflict,
                    false,
                    "Re-planned pass must succeed cleanly without conflict",
                );
                assert(
                    alphaAlerts.some((a) => a.code === "STALE_STATE_CONFLICT"),
                    "Must emit STALE_STATE_CONFLICT alert on 400 ValidationError",
                );
            } finally {
                daemonAlpha.client.getGalaxy = origGetGalaxy;
            }
        }

        // 8b: Simulate Daemon Alpha hard crash (`stop()`) and restart from `state_alpha.db` (SQLite)!
        const preCrashState = JSON.parse(
            JSON.stringify(daemonAlpha.getPersistedState()),
        );
        assert(preCrashState.human.dispatchedIdempotencyKeys.length > 0);
        assert(preCrashState.beliefTracker != null);
        daemonAlpha.stop();

        const daemonAlphaRestarted = registerDaemon(
            new LiveAdmiralDaemon({
                apiUrl,
                gameId,
                email: alphaEmail,
                password,
                checkpointPath: ckptPath,
                statePath: alphaDbPath,
                mode: "autonomous",
                cashReserveCredits: 150,
                neutralBorderPosture: "defend",
                maxRequestsPerSecond: 2,
                carrierGraceTicks: 2,
                allowEmergencyDefenseReserveDip: true,
                tickOffsetMs: 500,
                pollIntervalMs: 1000,
                deadlineSafetyMs: 2000,
            }),
        );
        await daemonAlphaRestarted.initialize();
        const restored = daemonAlphaRestarted.getPersistedState();
        assert.equal(
            restored.lastProcessedTick,
            preCrashState.lastProcessedTick,
        );
        assert.deepEqual(
            restored.seatMemory?.seenTick,
            preCrashState.seatMemory?.seenTick,
            "SQLite restart must restore SeatMemory.seenTick",
        );
        assert.equal(
            restored.beliefTracker?.lastTick,
            preCrashState.beliefTracker?.lastTick,
            "SQLite restart must restore BeliefTracker state",
        );

        // Verify `Idempotency-Key` deduplication prevents duplicate execution of an already-sent action!
        const dupAction: LiveAction = {
            type: "SetResearchQueue",
            queue: ["weapons", "hyperspace"],
        };
        const dupKey = buildIdempotencyKey(
            gameId,
            restored.lastProcessedTick,
            0,
            dupAction,
            "crash-retry-scope",
        );
        const keySet = new Set<string>([dupKey]);
        const dedupRes = await dispatchLiveActions({
            client: daemonAlphaRestarted.client,
            gameId,
            tick: restored.lastProcessedTick,
            actions: [dupAction],
            batchScope: "crash-retry-scope",
            maxRequestsPerSecond: 2,
            deadlineSafetyMs: 0,
            dispatchedKeys: keySet,
        });
        assert.equal(dedupRes.dispatchedCount, 0);
        assert.equal(dedupRes.skippedDuplicateCount, 1);

        // 8c: Mid-Game Attach Time-Machine Backfill (`GET /api/game/:gameId/galaxy?tick=T`) + 1-Cycle Lookahead Search!
        const daemonMidGameAttach = registerDaemon(
            new LiveAdmiralDaemon({
                apiUrl,
                gameId,
                email: alphaEmail,
                password,
                checkpointPath: ckptPath,
                statePath: attachDbPath,
                mode: "advisor",
                cashReserveCredits: 150,
                neutralBorderPosture: "defend",
                maxRequestsPerSecond: 2,
                carrierGraceTicks: 2,
                allowEmergencyDefenseReserveDip: true,
                tickOffsetMs: 500,
                pollIntervalMs: 1000,
                deadlineSafetyMs: 2000,
                backfillMaxTicks: 6,
                useSearchPlanner: true,
                searchPlans: 2,
                searchWorlds: 2,
            }),
        );
        const attachInfo = await daemonMidGameAttach.initialize();
        assert(
            attachInfo.tick >= 20,
            `Expected mid-game tick >= 20, got ${attachInfo.tick}`,
        );
        assert.equal(
            daemonMidGameAttach.getPersistedState().backfilledUpToTick,
            attachInfo.tick - 1,
            "Fresh mid-game attach must backfill historical ticks via Time Machine `?tick=T`",
        );
        scenariosVerified.push(
            "Scenario 8: Concurrent GUI 400 Recovery, SQLite Crash Restart, Idempotency Dedup & Time-Machine Backfill",
        );

        console.log(
            "[E2E] 10/10 All 8 Two-Hybrid end-to-end validation scenarios PASSED!",
        );
        return {
            gameId,
            alphaPlayerId,
            betaPlayerId,
            scenariosVerified,
        };
    } finally {
        for (const d of activeDaemons) {
            try {
                d.stop();
            } catch {
                // ignore cleanup errors
            }
        }
        if (socketServer) {
            await new Promise<void>((resolve) =>
                socketServer!.close(() => resolve()),
            );
        }
        if (server && server.listening) {
            await new Promise<void>((resolve) =>
                server!.close(() => resolve()),
            );
        }
        if (mongo) {
            try {
                await mongo.disconnect();
            } catch {
                // ignore
            }
        }
        stopMongoDockerContainer();
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }
}

if (require.main === module) {
    runTwoHybridE2eValidation()
        .then((summary) => {
            console.log(JSON.stringify(summary, null, 2));
            process.exit(0);
        })
        .catch((err) => {
            console.error("[E2E FAILED]", err);
            process.exit(1);
        });
}
