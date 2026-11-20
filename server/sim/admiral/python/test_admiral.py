"""End-to-end verification suite for the Admiral AI/RL stack."""

import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _THIS_DIR.parents[2]
_UPSTREAM_PY = _SERVER_DIR / "sim" / "python"

for p in (str(_THIS_DIR), str(_UPSTREAM_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import belief_data as bd  # noqa: E402
from admiral_agent import AdmiralAgent, AdmiralSearchPlanner  # noqa: E402
from admiral_env import AdmiralEnv  # noqa: E402
from admiral_net import AdmiralNet, build_admiral_inputs  # noqa: E402
from admiral_rollout import play_admiral_game  # noqa: E402
from admiral_train import build_dataset_from_rollouts, ppo_update  # noqa: E402
from belief_tracker import VISIBLE, BeliefTracker  # noqa: E402


class TestAdmiralStack(unittest.TestCase):
    def test_01_belief_conservation_and_attrition(self):
        with AdmiralEnv() as env:
            env.reset(admiral_seats=[0], seed=17, max_ticks=60, fog=True)
            static = bd.Static(env.header())
            tracker = BeliefTracker(static, seat=0)

            env.act_admiral(0, {})
            env.step()
            obs = env.observe_admiral(0)
            frame = obs["frame"]

            est = tracker.constrain_beliefs(frame, include_transit_share=False)
            owner = np.asarray(frame["owner"], dtype=np.int32)
            stats = np.asarray(frame["stats"], dtype=np.float32)
            car = frame["carriers"]
            c_owner = np.asarray(car["owner"], dtype=np.int32)
            c_ships = np.asarray(car["ships"], dtype=np.float32)

            for q in range(1, static.P):
                q_stars = owner == q
                if not q_stars.any():
                    continue
                vis_c_ships = float(
                    c_ships[(c_owner == q) & (c_ships >= 0)].sum()
                )
                total_believed = float(est["ships"][q_stars].sum()) + vis_c_ships
                self.assertAlmostEqual(
                    total_believed, float(stats[q, 1]), places=2
                )

            # Verify integer conservation across sampled BeliefWorlds
            worlds = tracker.sample_belief_worlds(
                frame, k_worlds=3, rng=np.random.default_rng(9)
            )
            self.assertEqual(len(worlds), 3)
            for w in worlds:
                w_ships = np.asarray(w["ships"], dtype=np.int32)
                for q in range(1, static.P):
                    q_stars = owner == q
                    if not q_stars.any():
                        continue
                    vis_c = int(
                        round(float(c_ships[(c_owner == q) & (c_ships >= 0)].sum()))
                    )
                    self.assertEqual(
                        int(w_ships[q_stars].sum()) + vis_c,
                        int(round(float(stats[q, 1]))),
                    )

            # Verify unseen off-screen battle attrition detection
            fake_next = json.loads(json.dumps(frame))
            fake_next["tick"] = frame["tick"] + 4
            # Player 2 loses 45 ships off-screen despite positive industry!
            fake_next["stats"][2][1] = max(0, frame["stats"][2][1] - 45)
            deltas = tracker.delta_features(fake_next)
            self.assertGreater(float(deltas[2, 3]), 0.3)

    def test_02_network_bounds_and_late_game_stability(self):
        with AdmiralEnv() as env:
            env.reset(admiral_seats=[0], seed=5, max_ticks=30, fog=True)
            static = bd.Static(env.header())
            obs = env.observe_admiral(0, truth_frame=True)

        # Inject extreme 100,000-ship late-game numbers to test log1p stability
        obs["frame"]["stats"][1][1] = 100_000
        obs["truthFrame"]["stats"][1][1] = 100_000

        np_in = build_admiral_inputs(
            static, obs["frame"], 0, own_facts=obs["ownFacts"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdmiralNet(len(static.techs), d=64, n_layers=2).to(device)
        batch = {
            k: torch.from_numpy(v[None]).to(device) for k, v in np_in.items()
        }
        res = model.act_or_evaluate(batch, sample=True)
        fog_v = float(res["policy_out"]["fog_value"][0].item())
        self.assertTrue(np.isfinite(fog_v))
        self.assertGreaterEqual(fog_v, -1.5)
        self.assertLessEqual(fog_v, 1.5)

    def test_03_cross_play_with_commander_and_sbd_recording(self):
        league1_dir = _SERVER_DIR / "data" / "league1"
        cmd_ckpts = sorted(glob.glob(str(league1_dir / "ema_*.pt")))
        if not cmd_ckpts:
            cmd_ckpts = sorted(glob.glob(str(league1_dir / "ckpt_*.pt")))

        opponents = {}
        if cmd_ckpts:
            opponents[2] = {"kind": "commander", "path": cmd_ckpts[-1]}

        with tempfile.TemporaryDirectory() as tmpdir:
            sbd_path = os.path.join(tmpdir, "crossplay.sbd")
            npz_path = Path(tmpdir) / "r0.npz"

            arrays = play_admiral_game(
                seed=101,
                seats=2,
                checkpoint=None,
                max_ticks=40,
                learner_seats=[0, 1],
                opponents=opponents,
                record=sbd_path,
                cycle_only=True,
            )
            np.savez_compressed(npz_path, **arrays)

            # Verify SBD1 file loads via upstream belief_data.load() & bd.Game
            self.assertTrue(os.path.exists(sbd_path))
            hdr, sbd_arrays = bd.load(sbd_path)
            self.assertEqual(len(hdr["stars"]), 200)
            self.assertIn("stars", sbd_arrays)
            self.assertIn("visibility", sbd_arrays)
            rec_game = bd.Game(sbd_path)
            v0 = rec_game.view(10, 0, rec_game.memory(0))
            self.assertEqual(len(v0["owner"]), 200)

            # Verify Factored PPO + Distributional Critic + Aux Belief update on GPU/CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AdmiralNet(len(hdr["technologies"])).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
            ds = build_dataset_from_rollouts([npz_path], model, device)
            stats = ppo_update(model, opt, ds, device, epochs=2, batch_size=8)
            for k, v in stats.items():
                self.assertTrue(np.isfinite(v), f"Non-finite stat {k}={v}")

    def test_04_search_planner_lookahead(self):
        with AdmiralEnv() as env:
            res = env.reset(
                admiral_seats=[0], seed=29, max_ticks=30, fog=True, cycle_only=True
            )
            static = bd.Static(env.header())
            model = AdmiralNet(len(static.techs), d=64, n_layers=2)
            agent = AdmiralAgent(model, static, sample=False)
            planner = AdmiralSearchPlanner(
                agent, n_plans=3, n_worlds=2, lookahead_ticks=8
            )

            # Advance to tick 1 (cycle start)
            env.act_admiral(0, {})
            res = env.step()
            self.assertIn(0, res["seats"])

            obs = env.observe_admiral(0)
            dec = planner.decide(env, obs, 0, seed=29)
            self.assertIn("budgetSplit", dec)
            self.assertIn("beliefs", dec)
            env.act_admiral(0, dec)
            res2 = env.step()
            self.assertGreater(res2["tick"], obs["tick"])

    def test_05_mini_admiral_train_cli_smoke(self):
        league1_dir = _SERVER_DIR / "data" / "league1"
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(_THIS_DIR / "admiral_train.py"),
                "--run-dir",
                tmpdir,
                "--iters",
                "2",
                "--games-per-iter",
                "2",
                "--workers",
                "1",
                "--max-ticks",
                "40",
                "--d-model",
                "64",
                "--n-layers",
                "2",
                "--snapshot-every",
                "1",
                "--record-every",
                "2",
                "--eval-every",
                "0",
            ]
            if league1_dir.is_dir():
                cmd.extend(["--commander-dir", str(league1_dir)])

            proc = subprocess.run(
                cmd, cwd=str(_SERVER_DIR), capture_output=True, text=True
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"admiral_train failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertTrue((Path(tmpdir) / "ckpt_2.pt").exists())
            self.assertTrue((Path(tmpdir) / "ema_2.pt").exists())
            self.assertTrue((Path(tmpdir) / "league.json").exists())
            self.assertTrue((Path(tmpdir) / "log.jsonl").exists())
            sbd_files = list((Path(tmpdir) / "belief").glob("*.sbd"))
            self.assertGreaterEqual(len(sbd_files), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
