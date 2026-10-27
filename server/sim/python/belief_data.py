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


class Game:
    """One recorded game, with per-player memory built tick by tick."""

    def __init__(self, path):
        self.path = str(path)
        self.header, a = load(path)
        h = self.header
        self.ticks = a["ticks"]
        self.T = len(self.ticks)
        self.S = len(h["stars"])
        self.P = len(h["players"])
        self.cycle = h["productionTicks"]
        self.speed = h["carrierSpeed"]

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
        self.techs = h["technologies"]
        self.levels = self.players[:, :, len(pf) - len(self.techs):]

        xy = np.array([[s["x"], s["y"]] for s in h["stars"]], np.float32)
        self.xy = xy
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        self.travel = d / self.speed  # ticks between stars
        self.home = np.array([s["homeStar"] for s in h["stars"]])
        self.hyper_range = np.array(h["hyperspaceRange"])
        self._paths = {}

    def carriers_at(self, t):
        return self.carriers[self.coff[t]:self.coff[t + 1]]

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

    def _hyperspace(self, t, q):
        i = self.techs.index("hyperspace") if "hyperspace" in self.techs else None
        return self.levels[t, q, i] if i is not None else 1

    def _threat(self, t, p, star_ships, rows, carrier_ships, cycles):
        """Ships of every enemy of p that could reach each star within
        `cycles` cycles, travelling along that enemy's own hyperspace range:
        garrisons from their stars, and carriers from where they are headed
        (or orbiting)."""
        out = np.zeros(self.S, np.float32)
        owner = self.owner[t]
        c_owner = rows[:, self.cf["owner"]].astype(int)
        dest = rows[:, self.cf["destination"]].astype(int)
        src = rows[:, self.cf["source"]].astype(int)
        limit = cycles * self.cycle
        for q in range(self.P):
            if q == p:
                continue
            mine = (owner == q) & (star_ships > 0)
            cs = (c_owner == q) & (carrier_ships > 0)
            if not mine.any() and not cs.any():
                continue
            path = self.paths(self._hyperspace(t, q))
            if mine.any():
                out += star_ships[mine] @ (path[mine] <= limit)
            if cs.any():
                at = np.where(dest[cs] >= 0, dest[cs], src[cs])
                ok = at >= 0
                cxy = rows[cs][:, [self.cf["x"], self.cf["y"]]]
                first = np.sqrt(((cxy - self.xy[np.maximum(at, 0)]) ** 2).sum(-1)) / self.speed
                eta = first[:, None] + path[np.maximum(at, 0)]
                out += (carrier_ships[cs] * ok) @ (eta <= limit)
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
        seen_tick, mem_ships, mem_infra, mem_extras = mem
        S, P = self.S, self.P
        owner = self.owner[t]
        own = owner == p
        enemy = (owner >= 0) & ~own
        v = self.vis[t, p]
        rows = self.carriers_at(t)
        c_owner = rows[:, self.cf["owner"]].astype(int)
        c_enemy = c_owner != p
        bit = 1 << p
        c_vis = (rows[:, self.cf["visible"]].astype(np.int64) & bit) > 0
        c_ships_vis = (rows[:, self.cf["shipsVisible"]].astype(np.int64) & bit) > 0

        big = self.travel.max() + 1
        to_own = np.where(own[None], self.travel, big).min(1) if own.any() else np.full(S, big)
        to_enemy = np.where(enemy[None], self.travel, big).min(1) if enemy.any() else np.full(S, big)

        # Within hyperspace range of an enemy's star, by that enemy's level.
        hs = np.array([self._hyperspace(t, q) for q in range(P)])
        enemy_range = self.hyper_range[np.clip(hs, 1, len(self.hyper_range)) - 1]
        d = self.travel * self.speed
        in_range = np.zeros(S, bool)
        if enemy.any():
            in_range = (d[enemy] <= enemy_range[owner[enemy]][:, None]).any(0)

        known_ships = np.where(seen_tick[t] >= 0, mem_ships[t], 0).astype(np.float32)
        known_carrier = np.where(c_vis & c_ships_vis, rows[:, self.cf["ships"]], 0).astype(np.float32)
        true_carrier = rows[:, self.cf["ships"]].astype(np.float32)

        dest = rows[:, self.cf["destination"]].astype(int)
        seen_in = np.zeros(S, np.float32)
        m = c_enemy & c_vis & c_ships_vis & (dest >= 0)
        np.add.at(seen_in, dest[m], rows[m, self.cf["ships"]])

        age = np.where(seen_tick[t] >= 0, (self.ticks[t] - seen_tick[t]) / self.cycle, NEVER)

        feats = np.stack([
            self.xy[:, 0] / 100.0, self.xy[:, 1] / 100.0,
            own, enemy, owner < 0,
            v != HIDDEN,
            seen_tick[t] >= 0,
            age,
            known_ships / SCALE_SHIPS,
            *(mem_infra[t].T / SCALE_INFRA),
            mem_extras[t, :, 1],
            mem_extras[t, :, 0],
            self.home >= 0,
            to_own / self.cycle, to_enemy / self.cycle,
            in_range,
            self._threat(t, p, known_ships, rows, known_carrier, 1) / SCALE_SHIPS,
            self._threat(t, p, known_ships, rows, known_carrier, 2) / SCALE_SHIPS,
            seen_in / SCALE_SHIPS,
        ], 1).astype(np.float32)

        true_ships = self.ships[t].astype(np.float32)
        targets = np.stack([
            true_ships / SCALE_SHIPS,
            *(self.infra[t].T / SCALE_INFRA),
            self._threat(t, p, true_ships, rows, true_carrier, 1) / SCALE_SHIPS,
            self._threat(t, p, true_ships, rows, true_carrier, 2) / SCALE_SHIPS,
            self._incoming(t, p) / SCALE_SHIPS,
        ], 1).astype(np.float32)

        # Seats ordered with the observer first.
        order = [p] + [q for q in range(P) if q != p]
        pl = self.players[t]
        stats = []
        for q in order:
            mine = owner == q
            cs = c_owner == q
            stats.append([
                q == p, pl[q, self.pf["defeated"]],
                mine.sum() / 10.0,
                (self.ships[t][mine].sum() + rows[cs, self.cf["ships"]].sum()) / SCALE_SHIPS,
                cs.sum() / 10.0,
                *(self.infra[t][mine].sum(0) / SCALE_INFRA / 10.0),
            ])
        player_feats = np.concatenate(
            [np.array(stats, np.float32), self.levels[t][order].astype(np.float32)], 1)
        player_targets = np.stack([
            pl[order, self.pf["credits"]],
            pl[order, self.pf["creditsSpecialists"]],
            pl[order, self.pf["researchingNow"]],
        ], 1).astype(np.int32)

        return dict(
            star_x=feats,
            star_y=targets,
            star_known=(v == VISIBLE),
            player_x=player_feats,
            player_y=player_targets,
            player_progress=self.progress[t][order],
            player_order=np.array(order, np.int8),
            # Each star's owner as an index into this sample's player order
            # (-1 = unowned).
            star_owner_seat=np.where(owner >= 0, np.argsort(order)[owner], -1).astype(np.int8),
        )


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
