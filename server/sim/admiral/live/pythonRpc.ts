import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";
import readline from "readline";
import { AdmiralObservation, BeliefWorld } from "../admiralEnv";
import { AdmiralDecision } from "../tacticalExecutor";
import {
    DiplomaticTier,
    NeutralBorderPosture,
    PersistedBeliefTrackerState,
} from "./types";

export interface PythonDecideResult {
    decision: AdmiralDecision;
    resLogits: number[];
    fogValue: number;
    trackerState: PersistedBeliefTrackerState;
}

export interface PythonSearchProposal {
    plans: AdmiralDecision[];
    worlds: BeliefWorld[];
    resLogits: number[];
    trackerState: PersistedBeliefTrackerState;
}

/**
 * Persistent stdio JSON-lines client wrapping `live/python/live_admiral_rpc.py`.
 */
export class PythonAdmiralBridge {
    private proc: ChildProcessWithoutNullStreams | null = null;
    private rl: readline.Interface | null = null;
    private nextId = 1;
    private pending = new Map<
        number,
        { resolve: (v: any) => void; reject: (e: Error) => void }
    >();
    private pythonBin: string;
    private scriptPath: string;

    constructor(pythonBin = process.env.PYTHON_BIN || "python3") {
        this.pythonBin = pythonBin;
        this.scriptPath = path.join(__dirname, "python", "live_admiral_rpc.py");
    }

    private ensureStarted() {
        if (this.proc && !this.proc.killed) return;

        const proc = spawn(this.pythonBin, ["-u", this.scriptPath], {
            stdio: ["pipe", "pipe", "pipe"],
        });
        this.proc = proc;

        const stderrChunks: string[] = [];
        proc.stderr.on("data", (buf) => {
            stderrChunks.push(buf.toString("utf8"));
            if (stderrChunks.length > 20) stderrChunks.shift();
        });

        this.rl = readline.createInterface({ input: proc.stdout });
        this.rl.on("line", (line) => {
            const trimmed = line.trim();
            if (!trimmed) return;
            let msg: any;
            try {
                msg = JSON.parse(trimmed);
            } catch {
                return;
            }
            const entry = this.pending.get(msg.id);
            if (!entry) return;
            this.pending.delete(msg.id);
            if (msg.ok) {
                entry.resolve(msg.result);
            } else {
                entry.reject(new Error(msg.error ?? "Python RPC error"));
            }
        });

        proc.on("exit", (code) => {
            const err = new Error(
                `Python Admiral RPC exited (${code}): ${stderrChunks.join("")}`,
            );
            for (const p of this.pending.values()) {
                p.reject(err);
            }
            this.pending.clear();
            this.proc = null;
            this.rl = null;
        });
    }

    private call<T = any>(cmd: string, args: Record<string, any>): Promise<T> {
        this.ensureStarted();
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.proc!.stdin.write(
                JSON.stringify({ id, cmd, args }) + "\n",
                "utf8",
            );
        });
    }

    async init(params: {
        header: any;
        seat: number;
        checkpoint?: string;
        trackerState?: PersistedBeliefTrackerState | null;
    }) {
        return this.call("init", params);
    }

    async ingestFrames(frames: any[]) {
        if (!frames.length) return { ingested: 0 };
        return this.call("ingest_frames", { frames });
    }

    async decide(params: {
        obs: AdmiralObservation;
        diplomaticTiers: DiplomaticTier[];
        neutralBorderPosture: NeutralBorderPosture;
        researchAllowedMask?: boolean[];
        bannedStarSpecialists?: number[];
        contestedUnownedStars?: number[];
    }): Promise<PythonDecideResult> {
        return this.call("decide", params);
    }

    async proposeSearch(params: {
        obs: AdmiralObservation;
        diplomaticTiers: DiplomaticTier[];
        neutralBorderPosture: NeutralBorderPosture;
        researchAllowedMask?: boolean[];
        bannedStarSpecialists?: number[];
        contestedUnownedStars?: number[];
        nPlans?: number;
        nWorlds?: number;
        seed?: number;
    }): Promise<PythonSearchProposal> {
        return this.call("propose_search", params);
    }

    async scoreLeaves(params: {
        leaves: {
            frame: any;
            ownFacts: any;
            players: { stars: number; ships: number; defeated: boolean }[];
            ticksToEnd: number | null;
        }[];
        diplomaticTiers: DiplomaticTier[];
        neutralBorderPosture: NeutralBorderPosture;
    }): Promise<{ scores: number[] }> {
        return this.call("score_leaves", params);
    }

    close() {
        if (this.rl) {
            this.rl.close();
            this.rl = null;
        }
        if (this.proc) {
            this.proc.stdin.end();
            this.proc.kill("SIGTERM");
            this.proc = null;
        }
        this.pending.clear();
    }
}
