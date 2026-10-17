// Save your current view of a live game to a JSON file.
//
//   npx tsx sim/fetchGalaxy.ts <gameId> <out.json>
//
// Authenticate with either:
//   SOLARIS_COOKIE   the Cookie header from a logged-in browser session, or
//   SOLARIS_EMAIL + SOLARIS_PASSWORD   to log in through /api/auth/login.
// SOLARIS_API overrides the API host (default https://api.solaris.games).
//
// This makes one or two requests per run. Keep it to your own games and
// about one run per tick.
import fs from "fs";

const api = process.env.SOLARIS_API ?? "https://api.solaris.games";

const login = async (): Promise<string> => {
    if (process.env.SOLARIS_COOKIE) {
        return process.env.SOLARIS_COOKIE;
    }

    const email = process.env.SOLARIS_EMAIL;
    const password = process.env.SOLARIS_PASSWORD;

    if (!email || !password) {
        throw new Error(
            "Set SOLARIS_COOKIE, or SOLARIS_EMAIL and SOLARIS_PASSWORD.",
        );
    }

    const res = await fetch(`${api}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
        throw new Error(`Login failed: ${res.status} ${await res.text()}`);
    }

    const cookies = res.headers.getSetCookie().map((c) => c.split(";")[0]);
    return cookies.join("; ");
};

const main = async () => {
    const [gameId, out] = process.argv.slice(2);

    if (!gameId || !out) {
        throw new Error("Usage: fetchGalaxy.ts <gameId> <out.json>");
    }

    const cookie = await login();
    const res = await fetch(`${api}/api/game/${gameId}/galaxy`, {
        headers: { Cookie: cookie },
    });

    if (!res.ok) {
        throw new Error(`Fetch failed: ${res.status} ${await res.text()}`);
    }

    const galaxy = await res.json();
    fs.writeFileSync(out, JSON.stringify(galaxy, null, 1));

    const viewer = galaxy.galaxy.players.find((p) => p.userId != null);
    console.log(
        `Saved tick ${galaxy.state.tick} as ${viewer?.alias ?? "spectator"}: ` +
            `${galaxy.galaxy.stars.length} stars, ${galaxy.galaxy.carriers.length} visible carriers.`,
    );
};

main().catch((e) => {
    console.error(e.message);
    process.exit(1);
});
