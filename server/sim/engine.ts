// In-memory game engine: create games and advance them tick by tick using
// the server's own tick logic.
//
// Games are plain objects (lean documents) rather than mongoose documents.
// The tick logic gives identical results either way and runs about five
// times faster on plain objects.
import fs from "fs";
import path from "path";
import { createSimContainer, SimGameModel } from "./container";
import { GameTickContext } from "../services/gameProcessing/context";
import { Game } from "../services/types/Game";
import { DBObjectId } from "../services/types/DBObjectId";
import { DependencyContainer } from "../services/types/DependencyContainer";
import { seededRng } from "./random";
import { AiConfig, DEFAULT_AI_CONFIG } from "../services/types/Ai";

const nullService: any = new Proxy({}, { get: () => async () => {} });

const toLean = (doc: any): Game => {
    const game = doc.toObject();
    game.save = async () => game;
    return game;
};

export class SimEngine {
    container: DependencyContainer;
    // Built-in AI tuning per player id. Players not listed use the defaults.
    aiConfigs = new Map<string, AiConfig>();

    constructor(container?: DependencyContainer) {
        this.container = container ?? createSimContainer();
        this.container.aiService.getConfig = (game, player) =>
            this.aiConfigs.get(player._id.toString()) ?? DEFAULT_AI_CONFIG;
    }

    // The server draws all of its randomness (combat ties, research
    // experiments, AI choices, map generation) from Math.random, so seeding
    // it makes a run reproducible.
    seed(seed: number) {
        Math.random = seededRng(seed);
    }

    loadSettings(name: string) {
        const file = path.join(
            __dirname,
            "../config/game/settings/official",
            `${name}.json`,
        );
        return JSON.parse(fs.readFileSync(file, "utf8"));
    }

    // Create a brand new, started game. Every slot is empty, so every player
    // is controlled by the built-in AI until it is given a userId.
    async createGame(settings: any): Promise<Game> {
        // With only AI players, ready-to-quit would end the game at once.
        settings.general.readyToQuit = "disabled";

        const doc = (await this.container.gameCreateService.create(
            nullService,
            settings,
            null,
        )) as unknown as Game;

        this.container.gameJoinService.assignNonUserPlayersToAI(doc, false);
        this.container.gameJoinService.startGame(doc);

        return toLean(doc);
    }

    // Load plain game JSON, e.g. a saved game or a determinized snapshot.
    // Ids may be strings; they are cast to ObjectIds and schema defaults are
    // filled in.
    fromJSON(json: any): Game {
        return toLean(new SimGameModel(json));
    }

    toJSON(game: Game): any {
        // JSON drops the save() stub along with any other functions.
        return JSON.parse(JSON.stringify(game));
    }

    clone(game: Game): Game {
        return this.fromJSON(this.toJSON(game));
    }

    // Mark a player as driven from outside the engine. The built-in AI only
    // plays players without a userId.
    setExternallyControlled(
        game: Game,
        playerId: DBObjectId,
        userId: DBObjectId,
    ) {
        const player = game.galaxy.players.find(
            (p) => p._id.toString() === playerId.toString(),
        )!;
        player.userId = userId;
    }

    // Tune the built-in AI for one player. Fields left out keep their
    // defaults.
    setAiConfig(
        game: Game,
        playerId: DBObjectId,
        overrides: Partial<AiConfig>,
    ) {
        this.aiConfigs.set(playerId.toString(), {
            ...DEFAULT_AI_CONFIG,
            ...overrides,
            starValueWeights: {
                ...DEFAULT_AI_CONFIG.starValueWeights,
                ...overrides.starValueWeights,
            },
            orderPriorities: {
                ...DEFAULT_AI_CONFIG.orderPriorities,
                ...overrides.orderPriorities,
            },
        });
    }

    async tick(game: Game, ticks: number = 1) {
        const tickService = this.container.gameTickService as any;

        for (let i = 0; i < ticks && !game.state.endDate; i++) {
            const context = new GameTickContext(
                game,
                [],
                nullService,
                nullService,
                nullService,
                nullService,
            );
            context.save = async () => {};

            await tickService._processTick(
                game,
                context,
                () => {},
                process.hrtime(),
            );
        }
    }

    // The JSON a player would get from GET /api/game/:gameId/galaxy, with
    // fog of war applied. Useful for testing the advisor against games where
    // the full state is known.
    async viewAs(game: Game, userId: DBObjectId): Promise<any> {
        const c = this.container as any;
        const service = c.gameGalaxyService;

        const restore = {
            getGameStateTick: c.gameService.getGameStateTick,
            getByIdGalaxyLean: c.gameService.getByIdGalaxyLean,
        };

        c.gameService.getGameStateTick = async () => game.state.tick;
        c.gameService.getByIdGalaxyLean = async () =>
            this.fromJSON(this.toJSON(game));

        try {
            return JSON.parse(
                JSON.stringify(await service.getGalaxy(game._id, userId, null)),
            );
        } finally {
            Object.assign(c.gameService, restore);
        }
    }
}
