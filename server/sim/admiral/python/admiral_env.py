"""Persistent Python client for `sim/admiral/server.ts`.

Supports both Admiral seats (`observe_admiral` / `act_admiral` /
`lookahead_admiral`) and existing Commander seats (`observe` / `act` /
`lookahead`) side by side in the same Node process, plus batched RPCs
(`observe_batch` / `act_batch`) and reusable `reset()` across games so worker
pools start Node only once per worker.
"""

import glob
import json
import os
import shutil
import subprocess

SERVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _find_node():
    """Prefer an NVM Node >=20 binary when present (non-interactive shells may
    otherwise fall back to /usr/bin/node v18), else `node` on PATH."""
    env_node = os.environ.get("SOLARIS_NODE")
    if env_node and os.path.exists(env_node):
        return env_node
    nvm_dir = os.path.expanduser("~/.nvm/versions/node")
    if os.path.isdir(nvm_dir):
        candidates = sorted(glob.glob(os.path.join(nvm_dir, "v*", "bin", "node")))
        if candidates:
            return candidates[-1]
    return shutil.which("node") or "node"


def _default_command(server_dir):
    node_bin = _find_node()
    for root in (server_dir, os.path.dirname(server_dir)):
        cli = os.path.join(root, "node_modules", "tsx", "dist", "cli.mjs")
        if os.path.exists(cli):
            return [node_bin, cli, "sim/admiral/server.ts"], False
    return ["npx", "tsx", "sim/admiral/server.ts"], os.name == "nt"


class AdmiralEnvError(RuntimeError):
    pass


class AdmiralEnv:
    def __init__(self, server_dir=SERVER_DIR, command=None):
        self._server_dir = server_dir
        self._custom_command = command
        self._next_id = 0
        self.cycle_only = False
        self._proc = None
        self._start()

    def _start(self):
        command = self._custom_command
        shell = False
        if command is None:
            command, shell = _default_command(self._server_dir)
        self._proc = subprocess.Popen(
            command,
            cwd=self._server_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=shell,
        )

    def _call(self, cmd, **args):
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        self._next_id += 1
        request = {"id": self._next_id, "cmd": cmd, "args": args}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise AdmiralEnvError(
                    "Admiral simulator exited unexpectedly; check stderr."
                )
            if line.startswith("{"):
                break

        reply = json.loads(line)
        if not reply["ok"]:
            raise AdmiralEnvError(reply["error"])
        return reply["result"]

    def reset(
        self,
        admiral_seats=None,
        commander_seats=None,
        agents=None,
        settings="standard",
        seed=1,
        ai_configs=None,
        max_ticks=3000,
        max_gap=4,
        fog=True,
        record=None,
        record_info=None,
        cycle_only=False,
    ):
        self.cycle_only = cycle_only
        return self._call(
            "reset",
            admiralSeats=list(admiral_seats or []),
            commanderSeats=list(commander_seats or []),
            agents=list(agents) if agents is not None else None,
            settings=settings,
            seed=int(seed),
            aiConfigs={int(k): v for k, v in (ai_configs or {}).items()},
            maxTicks=int(max_ticks),
            maxGap=int(max_gap),
            cycleOnly=bool(cycle_only),
            fog=bool(fog),
            record=record,
            recordInfo=record_info or {},
        )

    def header(self):
        return self._call("header")

    def observe(self, seat, view=False, frame=False, truth_frame=False):
        return self._call(
            "observe",
            seat=int(seat),
            view=bool(view),
            frame=bool(frame),
            truthFrame=bool(truth_frame),
        )

    def observe_admiral(self, seat, truth_frame=False):
        return self._call(
            "observeAdmiral",
            seat=int(seat),
            truthFrame=bool(truth_frame),
        )

    def observe_batch(self, items):
        return self._call("observeBatch", items=items)

    def act(self, seat, decision):
        return self._call("act", seat=int(seat), decision=decision)

    def act_admiral(self, seat, decision):
        return self._call("actAdmiral", seat=int(seat), decision=decision)

    def act_batch(self, items):
        return self._call("actBatch", items=items)

    def step(self):
        return self._call("step")

    def standings(self):
        return self._call("standings")

    def lookahead(self, seat, plans, ticks, seed=1):
        return self._call(
            "lookahead",
            seat=int(seat),
            plans=plans,
            ticks=int(ticks),
            seed=int(seed),
        )

    def lookahead_admiral(self, seat, plans, worlds, ticks, seed=1):
        return self._call(
            "lookaheadAdmiral",
            seat=int(seat),
            plans=plans,
            worlds=worlds,
            ticks=int(ticks),
            seed=int(seed),
        )

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                if self._proc.poll() is None:
                    self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            finally:
                if self._proc.stdout:
                    self._proc.stdout.close()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
