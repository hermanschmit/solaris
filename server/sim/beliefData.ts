// Records what every player could see on every tick of a game, next to the
// true state, as training data for a belief model: a model that guesses
// what fog of war hides from a player.
//
// Per tick it stores the true state of every star, carrier and player, and
// per player which stars and carriers were in its scanning range. Features
// such as a player's memory of stars it saw earlier, and targets such as
// the true ships at the stars it cannot see, are built from these in
// Python (sim/python/belief_data.py), so they can change without
// re-running games.
//
// A file is gzipped: the bytes "SBD1", a uint32 header length, a JSON
// header padded to 8 bytes, then each array's raw little-endian bytes at
// the offset the header gives (counted from the end of the padded header),
// each padded to 8 bytes. Shapes are in the header too.
import fs from "fs";
import zlib from "zlib";
import { SimEngine } from "./engine";
import { Game } from "../services/types/Game";
import { Player } from "../services/types/Player";

type TypedArray =
    | Int8Array
    | Uint8Array
    | Int16Array
    | Int32Array
    | Float32Array;

const DTYPES: Record<string, string> = {
    Int8Array: "i1",
    Uint8Array: "u1",
    Int16Array: "i2",
    Int32Array: "i4",
    Float32Array: "f4",
};

// Star visibility codes.
export const HIDDEN = 0;
export const VISIBLE = 1;
// In scanning range, but the ships are hidden (a nebula).
export const VISIBLE_NO_SHIPS = 2;

// Per star, per tick.
const STAR_FIELDS = [
    "owner",
    "ships",
    "economy",
    "industry",
    "science",
    "specialist",
    "warpGate",
];

// Per carrier, per tick. visible and shipsVisible are bit masks with one
// bit per player.
const CARRIER_FIELDS = [
    "id",
    "owner",
    "x",
    "y",
    "ships",
    "source",
    "destination",
    "specialist",
    "visible",
    "shipsVisible",
];

// Per player, per tick, before the per technology fields.
const PLAYER_FIELDS = [
    "credits",
    "creditsSpecialists",
    "researchingNow",
    "researchingNext",
    "defeated",
];

export class BeliefRecorder {
    playerIndex = new Map<string, number>();
    starIndex = new Map<string, number>();
    carrierIndex = new Map<string, number>();
    technologies: string[];
    header: any;

    ticks: number[] = [];
    stars: Int32Array[] = [];
    visibility: Uint8Array[] = [];
    carriers: Float32Array[] = [];
    carrierOffsets: number[] = [0];
    players: Int32Array[] = [];
    progress: Float32Array[] = [];

    constructor(
        public engine: SimEngine,
        game: Game,
        info: Record<string, any> = {},
    ) {
        const distance = engine.container.distanceService;

        game.galaxy.players.forEach((p, i) =>
            this.playerIndex.set(p._id.toString(), i),
        );
        game.galaxy.stars.forEach((s, i) =>
            this.starIndex.set(s._id.toString(), i),
        );

        if (game.galaxy.players.length > 16) {
            throw new Error(
                "Carrier visibility masks hold at most 16 players.",
            );
        }

        const startLevels = game.settings.technology.startingTechnologyLevel;
        this.technologies = Object.keys(startLevels).filter(
            (k) => /^[^_$]/.test(k) && startLevels[k] > 0,
        );

        // Hyperspace and scanning range by technology level, so ranges can
        // be worked out from levels alone.
        const levels = Array.from({ length: 30 }, (_, i) => i + 1);

        this.header = {
            version: 1,
            ...info,
            productionTicks: game.settings.galaxy.productionTicks,
            carrierSpeed: game.settings.specialGalaxy.carrierSpeed,
            hyperspaceRange: levels.map((l) =>
                distance.getHyperspaceDistance(game, l),
            ),
            scanningRange: levels.map((l) =>
                distance.getScanningDistance(game, l),
            ),
            players: game.galaxy.players.map((p) => ({ alias: p.alias })),
            technologies: this.technologies,
            stars: game.galaxy.stars.map((s) => ({
                x: s.location.x,
                y: s.location.y,
                homeStar: game.galaxy.players.findIndex(
                    (p) => p.homeStarId?.toString() === s._id.toString(),
                ),
                naturalResources: s.naturalResources,
                isNebula: Boolean(s.isNebula),
                isPulsar: Boolean(s.isPulsar),
            })),
            starFields: STAR_FIELDS,
            carrierFields: CARRIER_FIELDS,
            playerFields: [
                ...PLAYER_FIELDS,
                ...this.technologies.map((t) => `level.${t}`),
            ],
            winner: null,
        };
    }

    _owner(id: any): number {
        return id ? (this.playerIndex.get(id.toString()) ?? -1) : -1;
    }

    _star(id: any): number {
        return id ? (this.starIndex.get(id.toString()) ?? -1) : -1;
    }

    _carrier(id: any): number {
        const key = id.toString();
        let index = this.carrierIndex.get(key);

        if (index === undefined) {
            index = this.carrierIndex.size;
            this.carrierIndex.set(key, index);
        }

        return index;
    }

    capture(game: Game) {
        const c = this.engine.container;
        const stars = game.galaxy.stars;
        const players = game.galaxy.players;
        const started = c.gameStateService.isStarted(game);

        this.ticks.push(game.state.tick);

        const starRow = new Int32Array(stars.length * STAR_FIELDS.length);
        stars.forEach((s, i) => {
            starRow.set(
                [
                    this._owner(s.ownedByPlayerId),
                    Math.floor(s.shipsActual ?? s.ships ?? 0),
                    s.infrastructure?.economy ?? 0,
                    s.infrastructure?.industry ?? 0,
                    s.infrastructure?.science ?? 0,
                    s.specialistId ?? -1,
                    s.warpGate ? 1 : 0,
                ],
                i * STAR_FIELDS.length,
            );
        });
        this.stars.push(starRow);

        // What each player sees, by the same rules as the game's galaxy view
        // (services/gameGalaxy.ts).
        const visibility = new Uint8Array(players.length * stars.length);
        const carrierVisible = new Map<any, number>();
        const carrierShipsVisible = new Map<any, number>();

        players.forEach((p: Player, pi) => {
            if (p.defeated) return;

            const scanned = c.scanningService.calculateViewpointScanning(game, [
                p,
            ]);

            stars.forEach((s, si) => {
                const inRange =
                    (scanned.stars.has(s) &&
                        !scanned.unscannedWormHoles.has(s)) ||
                    (c.scanningService.isStarAlwaysVisible(s) && started);

                if (inRange) {
                    visibility[pi * stars.length + si] =
                        c.starService.canPlayersSeeStarShips(s, [p._id])
                            ? VISIBLE
                            : VISIBLE_NO_SHIPS;
                }
            });

            for (const carrier of scanned.carriers) {
                carrierVisible.set(
                    carrier,
                    (carrierVisible.get(carrier) ?? 0) | (1 << pi),
                );

                if (
                    c.carrierService.canPlayersSeeCarrierShips(
                        game,
                        [p],
                        carrier,
                    )
                ) {
                    carrierShipsVisible.set(
                        carrier,
                        (carrierShipsVisible.get(carrier) ?? 0) | (1 << pi),
                    );
                }
            }
        });
        this.visibility.push(visibility);

        const carrierRow = new Float32Array(
            game.galaxy.carriers.length * CARRIER_FIELDS.length,
        );
        game.galaxy.carriers.forEach((carrier, i) => {
            const waypoint = carrier.orbiting ? null : carrier.waypoints[0];

            carrierRow.set(
                [
                    this._carrier(carrier._id),
                    this._owner(carrier.ownedByPlayerId),
                    carrier.location.x,
                    carrier.location.y,
                    carrier.ships ?? 0,
                    waypoint
                        ? this._star(waypoint.source)
                        : this._star(carrier.orbiting),
                    waypoint ? this._star(waypoint.destination) : -1,
                    carrier.specialistId ?? -1,
                    carrierVisible.get(carrier) ?? 0,
                    carrierShipsVisible.get(carrier) ?? 0,
                ],
                i * CARRIER_FIELDS.length,
            );
        });
        this.carriers.push(carrierRow);
        this.carrierOffsets.push(
            this.carrierOffsets[this.carrierOffsets.length - 1] +
                game.galaxy.carriers.length,
        );

        const width = PLAYER_FIELDS.length + this.technologies.length;
        const playerRow = new Int32Array(players.length * width);
        const progressRow = new Float32Array(
            players.length * this.technologies.length,
        );
        players.forEach((p, pi) => {
            playerRow.set(
                [
                    Math.floor(p.credits ?? 0),
                    Math.floor(p.creditsSpecialists ?? 0),
                    this.technologies.indexOf(p.researchingNow),
                    this.technologies.indexOf(p.researchingNext),
                    p.defeated ? 1 : 0,
                    ...this.technologies.map((t) => p.research[t]?.level ?? 0),
                ],
                pi * width,
            );
            progressRow.set(
                this.technologies.map((t) => p.research[t]?.progress ?? 0),
                pi * this.technologies.length,
            );
        });
        this.players.push(playerRow);
        this.progress.push(progressRow);

        if (game.state.winner) {
            this.header.winner = this._owner(game.state.winner);
        }
    }

    _concat<T extends TypedArray>(rows: T[], Type: { new (n: number): T }): T {
        const out = new Type(rows.reduce((a, r) => a + r.length, 0));
        let offset = 0;
        for (const row of rows) {
            out.set(row as any, offset);
            offset += row.length;
        }
        return out;
    }

    save(file: string) {
        const T = this.ticks.length;
        const S = this.header.stars.length;
        const P = this.header.players.length;
        const arrays: [string, TypedArray, number[]][] = [
            ["ticks", Int32Array.from(this.ticks), [T]],
            [
                "stars",
                this._concat(this.stars, Int32Array),
                [T, S, STAR_FIELDS.length],
            ],
            [
                "visibility",
                this._concat(this.visibility, Uint8Array),
                [T, P, S],
            ],
            [
                "carriers",
                this._concat(this.carriers, Float32Array),
                [this.carrierOffsets[T], CARRIER_FIELDS.length],
            ],
            ["carrierOffsets", Int32Array.from(this.carrierOffsets), [T + 1]],
            [
                "players",
                this._concat(this.players, Int32Array),
                [T, P, this.header.playerFields.length],
            ],
            [
                "progress",
                this._concat(this.progress, Float32Array),
                [T, P, this.technologies.length],
            ],
        ];

        const pad = (n: number) => (n + 7) & ~7;
        let offset = 0;
        const header = {
            ...this.header,
            arrays: arrays.map(([name, data, shape]) => {
                const entry = {
                    name,
                    dtype: DTYPES[data.constructor.name],
                    shape,
                    offset,
                };
                offset += pad(data.byteLength);
                return entry;
            }),
        };

        const json = Buffer.from(JSON.stringify(header));
        const prefix = Buffer.alloc(8);
        prefix.write("SBD1", 0, "ascii");
        prefix.writeUInt32LE(json.length, 4);
        const parts: Buffer[] = [
            prefix,
            json,
            Buffer.alloc(pad(json.length) - json.length),
        ];

        for (const [, data] of arrays) {
            const bytes = Buffer.from(
                data.buffer,
                data.byteOffset,
                data.byteLength,
            );
            parts.push(bytes, Buffer.alloc(pad(bytes.length) - bytes.length));
        }

        fs.writeFileSync(
            file,
            zlib.gzipSync(Buffer.concat(parts), { level: 6 }),
        );
    }
}
