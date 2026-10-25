// Builds the server's dependency container for in-memory simulation.
// No database connection is opened: every service that would touch Mongo
// during a tick or game creation is stubbed out below.
import "./env";
import mongoose from "mongoose";
import createContainer from "../services/index";
import GameModel from "../db/models/Game";
import { Config } from "../config/types/Config";
import { DependencyContainer } from "../services/types/DependencyContainer";
import { logger } from "../utils/logging";

const simConfig: Config = {
    sessionSecret: "sim",
    sessionSecureCookies: false,
    connectionString: "",
    serverUrl: "",
    clientUrl: "",
    corsUrls: [],
    cacheEnabled: false,
    smtp: { enabled: false },
    paypal: { environment: "sandbox" },
    discord: {},
    frontend: {} as any,
};

// Socket.io stand-in: any property access or call is a no-op.
const nullSocketServer: any = new Proxy(function () {}, {
    get: () => nullSocketServer,
    apply: () => nullSocketServer,
});

// A Game model whose documents never hit the database.
export const SimGameModel: any = class extends (GameModel as any) {
    async save() {
        return this;
    }
};

// Fail fast if anything tries to reach the database.
mongoose.set("bufferCommands", false);

// Replace every method on a service instance with an async no-op. Used for
// services whose only job is persisting side effects (events, notifications,
// statistics, emails), which some game services call directly.
const silence = (service: any) => {
    let proto = Object.getPrototypeOf(service);

    while (proto && proto !== Object.prototype) {
        for (const key of Object.getOwnPropertyNames(proto)) {
            if (key !== "constructor" && typeof service[key] === "function") {
                service[key] = async () => {};
            }
        }

        proto = Object.getPrototypeOf(proto);
    }
};

export const createSimContainer = (): DependencyContainer => {
    const container = createContainer(
        simConfig,
        nullSocketServer,
        logger("sim") as any,
    );

    const c = container as any;

    silence(c.eventService);
    silence(c.notificationService);
    silence(c.statisticsService);
    silence(c.emailService);
    silence(c.broadcastService);

    c.gameLockService.isLockedInDatabase = async () => true;
    c.historyService.log = async () => {};
    c.initialGameStateService.storeStateFor = async () => {};
    c.gameCreateService.gameModel = SimGameModel;
    c.gameTickService._emitEvents = async () => {};
    c.userService.getGameUsers = async () => [];

    // Used when rendering a player's view of the galaxy.
    c.historyService.getHistoryMinimumTick = async () => 0;
    c.historyService.getHistoryByTick = async () => null;
    c.guildUserService.listUsersWithGuildTags = async () => [];

    // The server ends a game once only AI players remain (they count as
    // ready to quit). In the sim, AI players are real participants, so a
    // game ends only by conquest, countdown or a single survivor.
    c.gameService.isReadyToQuitImmediateEnd = (game) =>
        game.galaxy.players.every((p) => p.readyToQuit || p.defeated);
    c.leaderboardService.getLastManStanding = (game, leaderboard) => {
        const undefeated = game.galaxy.players.filter((p) => !p.defeated);

        if (undefeated.length === 1) {
            return undefeated[0];
        }

        if (undefeated.length === 0) {
            return leaderboard[0].player;
        }

        return null;
    };

    return container;
};
