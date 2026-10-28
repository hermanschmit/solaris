"""Belief-model training samples from games recorded by sim/beliefBatch.ts.

A sample is one player's view of the galaxy at one tick: what it can see
now, what it remembers seeing earlier, and the true values that fog of war
hides from it. Needs numpy.

    python3 belief_data.py <dir of .sbd files> <out.npz> [every] [players_per_tick]

Per star (S stars), the features (STAR_FEATURES) describe what the player
knows. Ownership is always public; ships and infrastructure are known only
for stars in scanning range now, or remembered from the last time they were.
The frontier features say how close each star is, in carrier travel time,
to the player's own stars and to the nearest enemy star, and how many enemy
ships the player knows of that could reach it within one or two cycles.

The targets (STAR_TARGETS) are the true values: ships and infrastructure at
every star, the true number of enemy ships that could reach it within one or
two cycles, and the enemy ships that actually arrive there within the next
cycle. star_known marks the stars whose ships the player sees now, where the
ship targets are trivial.

Per player (P seats, ordered so the observing player comes first), the
features are the public statistics and technology levels, and the targets
are the hidden credits, current research and research progress.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HIDDEN, VISIBLE, VISIBLE_NO_SHIPS = 0, 1, 2

STAR_FEATURES = [
    "x", "y",
    "owner_self", "owner_enemy", "owner_none",
    "visible",            # in scanning range now
    "seen_ever",
    "age",                # cycles since ships were last seen (0 = now, 1e3 = never)
    "ships", "economy", "industry", "science",  # now or last seen, 0 if never
    "has_specialist",
    "warp_gate",
    "home_star",
    "ticks_to_own",       # travel ticks to the nearest own star
    "ticks_to_enemy",     # travel ticks to the nearest enemy star
    "in_enemy_range",     # within an enemy's hyperspace range of its stars
    "known_threat_1c",    # enemy ships known to the player that could reach it in 1 cycle
    "known_threat_2c",
    "seen_carriers_in",   # enemy ships on visible carriers headed here
]

STAR_TARGETS = [
    "ships", "economy", "industry", "science",
    "threat_1c", "threat_2c",
    "incoming_1c",        # enemy ships that actually arrive within the next cycle
]

PLAYER_FEATURES = [
    "is_self", "defeated",
    "stars", "ships", "carriers", "economy", "industry", "science",
]

PLAYER_TARGETS = ["credits", "credits_specialists", "researching_now"]

SCALE_SHIPS = 100.0
SCALE_INFRA = 10.0
NEVER = 1e3


def load(path):
    """Read a .sbd file into (header, {name: array})."""
    raw = gzip.open(path, "rb").read()
    assert raw[:4] == b"SBD1", f"{path} is not a belief data file"
    n = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8:8 + n])
    base = 8 + ((n + 7) & ~7)
    arrays = {}
    for a in header["arrays"]:
        dtype = np.dtype(a["dtype"]).newbyteorder("<")
        count = int(np.prod(a["shape"]))
        arrays[a["name"]] = np.frombuffer(
            raw, dtype, count, base + a["offset"]).reshape(a["shape"])
    return header, arrays


class Static:
    """What never changes in a game: star positions, distances, ranges."""

    def __init__(self, header):
        h = self.header = header
        self.S = len(h["stars"])
        self.P = len(h["players"])
        self.cycle = h["productionTicks"]
        self.speed = h["carrierSpeed"]
        self.techs = h["technologies"]
        xy = np.array([[s["x"], s["y"]] for s in h["stars"]], np.float32)
        self.xy = xy
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        self.travel = d / self.speed  # ticks between stars
        self.home = np.array([s["homeStar"] for s in h["stars"]])
        self.hyper_range = np.array(h["hyperspaceRange"])
        self._hs = self.techs.index("hyperspace") if "hyperspace" in self.techs else None
        self._paths = {}

    def paths(self, level):
        """Travel ticks between every pair of stars for a carrier with the
        given hyperspace level: shortest routes whose hops are each within
        range. Unreachable pairs are inf. Warp gates are ignored."""
        level = int(np.clip(level, 1, len(self.hyper_range)))
        if level not in self._paths:
            r = self.hyper_range[level - 1] / self.speed
            m = np.where(self.travel <= r, self.travel, np.inf)
            for k in range(self.S):  # Floyd-Warshall
                np.minimum(m, m[:, k:k + 1] + m[k:k + 1, :], out=m)
            self._paths[level] = m
        return self._paths[level]

    def hyperspace(self, levels, q):
        return levels[q, self._hs] if self._hs is not None else 1

    def threat(self, p, owner, levels, star_ships, carriers, carrier_ships, cycles):
        """Ships of every enemy of p that could reach each star within
        `cycles` cycles, travelling along that enemy's own hyperspace range:
        garrisons from their stars, and carriers from where they are headed
        (or orbiting)."""
        out = np.zeros(self.S, np.float32)
        c_owner = carriers["owner"]
        dest, src = carriers["destination"], carriers["source"]
        limit = cycles * self.cycle
        for q in range(self.P):
            if q == p:
                continue
            mine = (owner == q) & (star_ships > 0)
            cs = (c_owner == q) & (carrier_ships > 0)
            if not mine.any() and not cs.any():
                continue
            path = self.paths(self.hyperspace(levels, q))
            if mine.any():
                out += star_ships[mine] @ (path[mine] <= limit)
            if cs.any():
                at = np.where(dest[cs] >= 0, dest[cs], src[cs])
                ok = at >= 0
                cxy = np.stack([carriers["x"][cs], carriers["y"][cs]], 1)
                first = np.sqrt(((cxy - self.xy[np.maximum(at, 0)]) ** 2).sum(-1)) / self.speed
                eta = first[:, None] + path[np.maximum(at, 0)]
                out += (carrier_ships[cs] * ok) @ (eta <= limit)
        return out

    def features(self, view, p):
        """Model inputs for player p from what it sees and remembers.

        view holds only what the player knows, as a seat view from the
        agent environment does (see AgentEnv.observe with fog) or as
        Game.view builds it from a recording:

          owner [S]          star owners (public), -1 unowned
          visibility [S]     HIDDEN, VISIBLE or VISIBLE_NO_SHIPS
          tick               the current tick
          seen_tick [S]      when each star's ships were last seen, -1 never
          mem_ships [S]      ships when last seen
          mem_infra [S, 3]   economy, industry, science when last in range
          mem_extras [S, 2]  warp gate, has specialist, when last in range
          carriers           visible carriers: dict of arrays owner, x, y,
                             ships (-1 when hidden), source, destination
          levels [P, T]      every player's technology levels (public)
          stats [P, 6]       every player's stars, ships, carriers, economy,
                             industry, science (public)
          defeated [P]
        """
        S, P = self.S, self.P
        owner = np.asarray(view["owner"])
        v = np.asarray(view["visibility"])
        seen_tick = np.asarray(view["seen_tick"])
        levels = np.asarray(view["levels"])
        stats = np.asarray(view["stats"], np.float32)
        car = {k: np.asarray(x) for k, x in view["carriers"].items()}
        own = owner == p
        enemy = (owner >= 0) & ~own

        big = self.travel.max() + 1
        to_own = np.where(own[None], self.travel, big).min(1) if own.any() else np.full(S, big)
        to_enemy = np.where(enemy[None], self.travel, big).min(1) if enemy.any() else np.full(S, big)

        # Within hyperspace range of an enemy's star, by that enemy's level.
        hs = np.array([self.hyperspace(levels, q) for q in range(P)])
        enemy_range = self.hyper_range[np.clip(hs, 1, len(self.hyper_range)) - 1]
        d = self.travel * self.speed
        in_range = np.zeros(S, bool)
        if enemy.any():
            in_range = (d[enemy] <= enemy_range[owner[enemy]][:, None]).any(0)

        known_ships = np.where(seen_tick >= 0, view["mem_ships"], 0).astype(np.float32)
        known_carrier = np.maximum(car["ships"], 0).astype(np.float32)

        dest = car["destination"].astype(int)
        seen_in = np.zeros(S, np.float32)
        m = (car["owner"] != p) & (car["ships"] >= 0) & (dest >= 0)
        np.add.at(seen_in, dest[m], car["ships"][m])

        age = np.where(seen_tick >= 0, (view["tick"] - seen_tick) / self.cycle, NEVER)
        mem_infra = np.asarray(view["mem_infra"], np.float32)
        mem_extras = np.asarray(view["mem_extras"], np.float32)

        feats = np.stack([
            self.xy[:, 0] / 100.0, self.xy[:, 1] / 100.0,
            own, enemy, owner < 0,
            v != HIDDEN,
            seen_tick >= 0,
            age,
            known_ships / SCALE_SHIPS,
            *(mem_infra.T / SCALE_INFRA),
            mem_extras[:, 1],
            mem_extras[:, 0],
            self.home >= 0,
            to_own / self.cycle, to_enemy / self.cycle,
            in_range,
            self.threat(p, owner, levels, known_ships, car, known_carrier, 1) / SCALE_SHIPS,
            self.threat(p, owner, levels, known_ships, car, known_carrier, 2) / SCALE_SHIPS,
            seen_in / SCALE_SHIPS,
        ], 1).astype(np.float32)

        # Seats ordered with the observer first.
        order = [p] + [q for q in range(P) if q != p]
        player = np.concatenate([
            np.array([[q == p] for q in order], np.float32),
            np.asarray(view["defeated"], np.float32)[order, None],
            stats[order, 0:1] / 10.0,
            stats[order, 1:2] / SCALE_SHIPS,
            stats[order, 2:3] / 10.0,
            stats[order, 3:6] / SCALE_INFRA / 10.0,
            levels[order].astype(np.float32),
        ], 1)

        return dict(
            star_x=feats,
            player_x=player,
            player_order=np.array(order, np.int8),
            # Each star's owner as an index into this sample's player order
            # (-1 = unowned).
            star_owner_seat=np.where(owner >= 0, np.argsort(order)[owner], -1).astype(np.int8),
        )


class Game(Static):
    """One recorded game, with per-player memory built tick by tick."""

    def __init__(self, path):
        self.path = str(path)
        header, a = load(path)
        super().__init__(header)
        h = header
        self.ticks = a["ticks"]
        self.T = len(self.ticks)

        sf = {f: i for i, f in enumerate(h["starFields"])}
        st = a["stars"]
        self.owner = st[:, :, sf["owner"]]
        self.ships = st[:, :, sf["ships"]]
        self.infra = st[:, :, [sf["economy"], sf["industry"], sf["science"]]]
        self.specialist = st[:, :, sf["specialist"]]
        self.warp = st[:, :, sf["warpGate"]]
        self.vis = a["visibility"]

        cf = {f: i for i, f in enumerate(h["carrierFields"])}
        self.cf = cf
        self.carriers = a["carriers"]
        self.coff = a["carrierOffsets"]

        pf = {f: i for i, f in enumerate(h["playerFields"])}
        self.pf = pf
        self.players = a["players"]
        self.progress = a["progress"]
        self.levels = self.players[:, :, len(pf) - len(self.techs):]

    def carriers_at(self, t):
        return self.carriers[self.coff[t]:self.coff[t + 1]]

    def _carrier_dict(self, rows):
        cf = self.cf
        return {k: rows[:, cf[k]].astype(np.float32 if k in ("x", "y", "ships") else int)
                for k in ("owner", "x", "y", "ships", "source", "destination")}

    def memory(self, p):
        """For player p, per tick: the tick each star's ships were last seen
        (-1 = never), and the last-seen ships, infrastructure, and warp gate
        and specialist flags."""
        T, S = self.T, self.S
        seen_tick = np.full((T, S), -1, np.int32)
        ships = np.zeros((T, S), np.float32)
        infra = np.zeros((T, S, 3), np.float32)
        # Warp gates and specialists, as last seen in scanning range.
        extras = np.zeros((T, S, 2), np.float32)
        cur_extras = np.zeros((S, 2), np.float32)
        cur_tick = np.full(S, -1, np.int32)
        cur_ships = np.zeros(S, np.float32)
        cur_infra = np.zeros((S, 3), np.float32)
        for t in range(T):
            v = self.vis[t, p]
            inrange = v != HIDDEN
            full = v == VISIBLE
            cur_infra[inrange] = self.infra[t][inrange]
            cur_extras[inrange, 0] = self.warp[t][inrange] > 0
            cur_extras[inrange, 1] = self.specialist[t][inrange] >= 0
            cur_ships[full] = self.ships[t][full]
            cur_tick[full] = self.ticks[t]
            seen_tick[t], ships[t], infra[t] = cur_tick, cur_ships, cur_infra
            extras[t] = cur_extras
        return seen_tick, ships, infra, extras

    def view(self, t, p, mem):
        """What player p knows at tick index t, in the form features() takes."""
        seen_tick, mem_ships, mem_infra, mem_extras = mem
        rows = self.carriers_at(t)
        bit = 1 << p
        vis = (rows[:, self.cf["visible"]].astype(np.int64) & bit) > 0
        ships_vis = (rows[:, self.cf["shipsVisible"]].astype(np.int64) & bit) > 0
        seen = self._carrier_dict(rows[vis])
        seen["ships"] = np.where(ships_vis[vis], seen["ships"], -1).astype(np.float32)
        return dict(
            tick=int(self.ticks[t]),
            owner=self.owner[t],
            visibility=self.vis[t, p],
            seen_tick=seen_tick[t],
            mem_ships=mem_ships[t],
            mem_infra=mem_infra[t],
            mem_extras=mem_extras[t],
            carriers=seen,
            levels=self.levels[t],
            stats=self.public_stats(t),
            defeated=self.players[t, :, self.pf["defeated"]],
        )

    def public_stats(self, t):
        """Every player's stars, ships, carriers, economy, industry and
        science, which the game shows to everyone."""
        owner = self.owner[t]
        rows = self.carriers_at(t)
        c_owner = rows[:, self.cf["owner"]].astype(int)
        out = np.zeros((self.P, 6), np.float32)
        for q in range(self.P):
            mine = owner == q
            cs = c_owner == q
            out[q] = [mine.sum(),
                      self.ships[t][mine].sum() + rows[cs, self.cf["ships"]].sum(),
                      cs.sum(), *self.infra[t][mine].sum(0)]
        return out

    def _incoming(self, t, p):
        """Enemy ships on carriers headed for each star, at their most over
        the next cycle (a carrier counts while it is heading there)."""
        out = np.zeros(self.S, np.float32)
        for u in range(t, min(t + self.cycle, self.T)):
            rows = self.carriers_at(u)
            enemy = (rows[:, self.cf["owner"]] != p) & (rows[:, self.cf["destination"]] >= 0)
            per = np.zeros(self.S, np.float32)
            np.add.at(per, rows[enemy, self.cf["destination"]].astype(int),
                      rows[enemy, self.cf["ships"]])
            out = np.maximum(out, per)
        return out

    def sample(self, t, p, mem):
        sample = self.features(self.view(t, p, mem), p)
        order = sample["player_order"].astype(int)

        owner = self.owner[t]
        levels = self.levels[t]
        car = self._carrier_dict(self.carriers_at(t))
        true_ships = self.ships[t].astype(np.float32)
        sample["star_y"] = np.stack([
            true_ships / SCALE_SHIPS,
            *(self.infra[t].T / SCALE_INFRA),
            self.threat(p, owner, levels, true_ships, car, car["ships"], 1) / SCALE_SHIPS,
            self.threat(p, owner, levels, true_ships, car, car["ships"], 2) / SCALE_SHIPS,
            self._incoming(t, p) / SCALE_SHIPS,
        ], 1).astype(np.float32)
        sample["star_known"] = self.vis[t, p] == VISIBLE

        pl = self.players[t]
        sample["player_y"] = np.stack([
            pl[order, self.pf["credits"]],
            pl[order, self.pf["creditsSpecialists"]],
            pl[order, self.pf["researchingNow"]],
        ], 1).astype(np.int32)
        sample["player_progress"] = self.progress[t][order]
        return sample


def build(files, every=10, per_tick=3, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    meta = []
    for f in files:
        g = Game(f)
        alive_tick = lambda t, p: g.players[t, p, g.pf["defeated"]] == 0
        mems = {}
        for t in range(g.cycle, g.T, every):
            alive = [p for p in range(g.P) if alive_tick(t, p)]
            for p in rng.choice(alive, min(per_tick, len(alive)), replace=False):
                p = int(p)
                if p not in mems:
                    mems[p] = g.memory(p)
                s = g.sample(t, p, mems[p])
                for k, v in s.items():
                    out.setdefault(k, []).append(v)
                meta.append((g.header["seed"], int(g.ticks[t]), p))
    arrays = {k: np.stack(v) for k, v in out.items()}
    arrays["meta"] = np.array(meta, np.int32)
    return arrays


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    per_tick = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    files = sorted(Path(src).glob("*.sbd"))
    arrays = build(files, every, per_tick)
    np.savez_compressed(dst, star_features=np.array(STAR_FEATURES),
                        star_targets=np.array(STAR_TARGETS), **arrays)
    print(f"{len(files)} games, {len(arrays['meta'])} samples -> {dst}")
