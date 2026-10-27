"""Python client for the Solaris simulator's agent environment.

Starts `npx tsx sim/server.ts` as a subprocess and talks to it over JSON
lines. Needs only the standard library.

    from solaris_env import SolarisEnv

    with SolarisEnv() as env:
        result = env.reset(agents=[0], seed=1)
        while not result["done"]:
            for seat in result["seats"]:
                obs = env.observe(seat)
                env.act(seat, {"orders": {c["key"]: c["score"] for c in obs["candidates"]}})
            result = env.step()
        print(env.standings())
"""

import json
import os
import subprocess

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class SolarisError(RuntimeError):
    pass


class SolarisEnv:
    def __init__(self, server_dir=SERVER_DIR, command=None):
        self._next_id = 0
        self._proc = subprocess.Popen(
            command or ["npx", "tsx", "sim/server.ts"],
            cwd=server_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            # On Windows, npx is a batch file and needs a shell.
            shell=(os.name == "nt" and command is None),
        )

    def _call(self, cmd, **args):
        self._next_id += 1
        request = {"id": self._next_id, "cmd": cmd, "args": args}
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise SolarisError("The simulator exited; see its output above.")
            if line.startswith("{"):
                break

        reply = json.loads(line)
        if not reply["ok"]:
            raise SolarisError(reply["error"])
        return reply["result"]

    def reset(self, agents, settings="standard", seed=1, ai_configs=None, max_ticks=3000, max_gap=4):
        """Start a new game. `agents` lists the seats (0 to 9) you control.
        Returns {"tick", "done", "seats"}: the seats that must decide now."""
        return self._call(
            "reset",
            agents=agents,
            settings=settings,
            seed=seed,
            aiConfigs=ai_configs or {},
            maxTicks=max_ticks,
            maxGap=max_gap,
        )

    def observe(self, seat, view=False):
        """The seat's candidate orders, with the built-in AI's scores. With
        view=True, also its fog-of-war galaxy view (slower)."""
        return self._call("observe", seat=seat, view=view)

    def act(self, seat, decision):
        """Set the seat's decision: {"orders": {key: score}, "config": {...},
        "research": "weapons"}. All fields are optional."""
        return self._call("act", seat=seat, decision=decision)

    def step(self):
        """Advance until a seat needs a decision or the game ends."""
        return self._call("step")

    def standings(self):
        return self._call("standings")

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
