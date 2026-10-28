// Fog of war for agent seats: what a seat has seen and remembers, and a
// lens that lets the built-in executor see only that.
//
// The built-in AI reads the whole game. For an agent seat under fog, the
// lens hides what the seat cannot see for as long as the executor runs:
// carriers out of scanning range are taken out of the galaxy, visible enemy
// carriers show only their next waypoint, and stars whose ships the seat
// cannot see show believed ships (from the agent's belief model, or else
// the last ships seen there) and the infrastructure last seen. Everything
// is put back afterwards, so the game itself never changes.
import { SimEngine } from "./engine";
import { Game } from "../services/types/Game";
import { Player } from "../services/types/Player";
import { Carrier } from "../services/types/Carrier";
import { seatView, SeatView, HIDDEN, VISIBLE } from "./beliefData";

// What a seat remembers of each star, updated every tick.
export class SeatMemory {
    // Tick the star's ships were last seen, -1 if never.
    seenTick: Int32Array;
    ships: Float32Array;
    // Economy, industry and science, per star, when last in scanning range.
    infrastructure: Float32Array;
    warpGate: Uint8Array;
    specialist: Uint8Array;
    view: SeatView | null = null;

    constructor(stars: number) {
        this.seenTick = new Int32Array(stars).fill(-1);
        this.ships = new Float32Array(stars);
        this.infrastructure = new Float32Array(stars * 3);
        this.warpGate = new Uint8Array(stars);
        this.specialist = new Uint8Array(stars);
    }

    update(engine: SimEngine, game: Game, player: Player) {
        const view = seatView(engine, game, player);
        this.view = view;

        game.galaxy.stars.forEach((s, i) => {
            if (view.stars[i] === HIDDEN) return;

            this.infrastructure.set(
                [
                    s.infrastructure?.economy ?? 0,
                    s.infrastructure?.industry ?? 0,
                    s.infrastructure?.science ?? 0,
                ],
                i * 3,
            );
            this.warpGate[i] = s.warpGate ? 1 : 0;
            this.specialist[i] = s.specialistId != null ? 1 : 0;

            if (view.stars[i] === VISIBLE) {
                this.ships[i] = Math.floor(s.shipsActual ?? s.ships ?? 0);
                this.seenTick[i] = game.state.tick;
            }
        });
    }
}

// What a seat knows right now, in the form sim/python/belief_data.py's
// Static.features takes. Star and player references are indexes.
export const seatFrame = (
    game: Game,
    memory: SeatMemory,
    technologies: string[],
) => {
    const view = memory.view!;
    const players = game.galaxy.players;
    const playerIndex = new Map(
        players.map((p, i) => [p._id.toString(), i] as [string, number]),
    );
    const starIndex = new Map(
        game.galaxy.stars.map(
            (s, i) => [s._id.toString(), i] as [string, number],
        ),
    );
    const ownerOf = (id: any) =>
        id ? (playerIndex.get(id.toString()) ?? -1) : -1;
    const starOf = (id: any) =>
        id ? (starIndex.get(id.toString()) ?? -1) : -1;

    // Public statistics: stars, ships, carriers, economy, industry, science.
    const stats = players.map(() => [0, 0, 0, 0, 0, 0]);
    for (const s of game.galaxy.stars) {
        const row = stats[ownerOf(s.ownedByPlayerId)];
        if (!row) continue;
        row[0] += 1;
        row[1] += Math.floor(s.shipsActual ?? s.ships ?? 0);
        row[3] += s.infrastructure?.economy ?? 0;
        row[4] += s.infrastructure?.industry ?? 0;
        row[5] += s.infrastructure?.science ?? 0;
    }
    for (const c of game.galaxy.carriers) {
        const row = stats[ownerOf(c.ownedByPlayerId)];
        if (!row) continue;
        row[1] += c.ships ?? 0;
        row[2] += 1;
    }

    const carriers = {
        owner: [] as number[],
        x: [] as number[],
        y: [] as number[],
        ships: [] as number[],
        source: [] as number[],
        destination: [] as number[],
    };
    for (const c of game.galaxy.carriers) {
        if (!view.carriers.has(c)) continue;
        const waypoint = c.orbiting ? null : c.waypoints[0];
        carriers.owner.push(ownerOf(c.ownedByPlayerId));
        carriers.x.push(c.location.x);
        carriers.y.push(c.location.y);
        carriers.ships.push(view.carrierShips.has(c) ? (c.ships ?? 0) : -1);
        carriers.source.push(
            waypoint ? starOf(waypoint.source) : starOf(c.orbiting),
        );
        carriers.destination.push(waypoint ? starOf(waypoint.destination) : -1);
    }

    const S = game.galaxy.stars.length;
    return {
        tick: game.state.tick,
        owner: game.galaxy.stars.map((s) => ownerOf(s.ownedByPlayerId)),
        visibility: Array.from(view.stars),
        seen_tick: Array.from(memory.seenTick),
        mem_ships: Array.from(memory.ships),
        mem_infra: Array.from({ length: S }, (_, i) =>
            Array.from(memory.infrastructure.subarray(i * 3, i * 3 + 3)),
        ),
        mem_extras: Array.from({ length: S }, (_, i) => [
            memory.warpGate[i],
            memory.specialist[i],
        ]),
        carriers,
        levels: players.map((p) =>
            technologies.map((t) => (p.research as any)[t]?.level ?? 0),
        ),
        stats,
        defeated: players.map((p) => (p.defeated ? 1 : 0)),
    };
};

// Make the game show only what the seat knows, until the returned function
// is called to put it back. believedShips gives the ships to show at a star
// whose ships the seat cannot see, by star index.
export const applyFog = (
    game: Game,
    player: Player,
    memory: SeatMemory,
    believedShips: (star: number) => number | undefined,
): (() => void) => {
    const view = memory.view!;
    const playerId = player._id.toString();
    const savedStars: [any, any, any, any][] = [];
    const savedWaypoints: [Carrier, any][] = [];
    const carriers = game.galaxy.carriers;
    const original = new Set(carriers);

    game.galaxy.stars.forEach((s, i) => {
        if (s.ownedByPlayerId?.toString() === playerId) return;
        if (view.stars[i] === VISIBLE) return;

        savedStars.push([s, s.ships, s.shipsActual, s.infrastructure]);
        const ships = Math.max(0, believedShips(i) ?? memory.ships[i]);
        s.ships = Math.floor(ships);
        s.shipsActual = ships;

        if (view.stars[i] === HIDDEN) {
            s.infrastructure = {
                economy: memory.infrastructure[i * 3],
                industry: memory.infrastructure[i * 3 + 1],
                science: memory.infrastructure[i * 3 + 2],
            };
        }
    });

    game.galaxy.carriers = carriers.filter((c) => {
        if (c.ownedByPlayerId?.toString() === playerId) return true;
        if (!view.carriers.has(c)) return false;

        if (c.waypoints.length > 1) {
            savedWaypoints.push([c, c.waypoints]);
            c.waypoints = c.waypoints.slice(0, 1);
        }
        return true;
    });

    return () => {
        // Carriers the executor bought are kept, after the originals, which
        // is where the game would have put them.
        const added = game.galaxy.carriers.filter((c) => !original.has(c));
        game.galaxy.carriers = carriers.concat(added);

        for (const [c, waypoints] of savedWaypoints) c.waypoints = waypoints;
        for (const [s, ships, shipsActual, infrastructure] of savedStars) {
            s.ships = ships;
            s.shipsActual = shipsActual;
            s.infrastructure = infrastructure;
        }
    };
};
