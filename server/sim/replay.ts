// Records a game tick by tick into a compact replay file that
// sim/viewer.html can play back in a browser.
//
// Star positions are stored once. Each frame stores, per star, its owner and
// ships, and per carrier its owner, position, ships and destination. Owners
// and destinations are indexes into the players and stars lists (-1 for
// none), which keeps a full game to a few megabytes.
import fs from "fs";
import { Game } from "../services/types/Game";

export interface ReplayFrame {
    tick: number;
    // Per star: owner index, or -1 when unowned.
    owners: number[];
    // Per star: garrisoned ships.
    ships: number[];
    // Per carrier, flattened: owner, x, y, ships, destination star (-1 when
    // orbiting).
    carriers: number[];
    // Per player, flattened: stars, total ships, economy, industry, science,
    // defeated (0 or 1).
    players: number[];
    // Per player, flattened: the level of each technology in
    // Replay.technologies, then the index of the one being researched (-1
    // when none). Missing from replays recorded before it was added.
    research?: number[];
}

export interface Replay {
    version: 1;
    settings: string;
    seed: number | null;
    productionTicks: number;
    players: { alias: string; colour: string }[];
    stars: { name: string; x: number; y: number }[];
    // Technologies enabled in this game, in the order frames store them.
    technologies?: string[];
    frames: ReplayFrame[];
    winner: number | null;
}

export class ReplayRecorder {
    replay: Replay;
    playerIndex = new Map<string, number>();
    starIndex = new Map<string, number>();

    constructor(game: Game, settings: string, seed: number | null = null) {
        game.galaxy.players.forEach((p, i) =>
            this.playerIndex.set(p._id.toString(), i),
        );
        game.galaxy.stars.forEach((s, i) =>
            this.starIndex.set(s._id.toString(), i),
        );

        this.replay = {
            version: 1,
            settings,
            seed,
            productionTicks: game.settings.galaxy.productionTicks,
            players: game.galaxy.players.map((p) => ({
                alias: p.alias,
                colour: p.colour.value,
            })),
            stars: game.galaxy.stars.map((s) => ({
                name: s.name,
                x: Math.round(s.location.x),
                y: Math.round(s.location.y),
            })),
            technologies: Object.keys(
                game.settings.technology.startingTechnologyLevel,
            ).filter(
                (k) =>
                    /^[^_$]/.test(k) &&
                    game.settings.technology.startingTechnologyLevel[k] > 0,
            ),
            frames: [],
            winner: null,
        };
    }

    _owner(id: any): number {
        return id ? (this.playerIndex.get(id.toString()) ?? -1) : -1;
    }

    capture(game: Game) {
        const stars = game.galaxy.stars;
        const totals = game.galaxy.players.map(() => [0, 0, 0, 0, 0]);

        const owners = stars.map((s) => this._owner(s.ownedByPlayerId));
        const ships = stars.map((s) =>
            Math.floor(s.shipsActual ?? s.ships ?? 0),
        );

        stars.forEach((s, i) => {
            const t = totals[owners[i]];
            if (!t) return;
            t[0] += 1;
            t[1] += ships[i];
            t[2] += s.infrastructure.economy ?? 0;
            t[3] += s.infrastructure.industry ?? 0;
            t[4] += s.infrastructure.science ?? 0;
        });

        const carriers: number[] = [];

        for (const c of game.galaxy.carriers) {
            const owner = this._owner(c.ownedByPlayerId);
            const destination =
                c.orbiting || !c.waypoints.length
                    ? -1
                    : (this.starIndex.get(
                          c.waypoints[0].destination.toString(),
                      ) ?? -1);

            carriers.push(
                owner,
                Math.round(c.location.x),
                Math.round(c.location.y),
                c.ships ?? 0,
                destination,
            );

            if (totals[owner]) totals[owner][1] += c.ships ?? 0;
        }

        const technologies = this.replay.technologies!;

        this.replay.frames.push({
            tick: game.state.tick,
            owners,
            ships,
            carriers,
            players: game.galaxy.players.flatMap((p, i) => [
                ...totals[i],
                p.defeated ? 1 : 0,
            ]),
            research: game.galaxy.players.flatMap((p) => [
                ...technologies.map((t) => p.research[t]?.level ?? 0),
                technologies.indexOf(p.researchingNow),
            ]),
        });

        if (game.state.winner) {
            this.replay.winner = this._owner(game.state.winner);
        }
    }

    save(file: string) {
        fs.writeFileSync(file, JSON.stringify(this.replay));
    }
}
