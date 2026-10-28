"""An agent seat that plays under fog of war with a belief model.

    python3 belief_agent.py <model.pt> [mode] [seed] [seat] [max_ticks]

mode is one of
  belief      fog on; the executor acts on the belief model's ship estimates
  memory      fog on; the executor assumes the last ships it saw
  omniscient  fog off; the executor reads the whole game (the built-in AI)

At each decision the agent turns what its seat sees and remembers into the
belief model's inputs (the same code that built its training data), runs
the model, and passes the believed ships at every star to the executor. It
keeps the built-in AI's choice of orders. It prints its stars over time
and its final placement, so the modes can be compared on the same seeds.
"""

import sys

import torch

import belief_data as bd
import belief_train as bt
from solaris_env import SolarisEnv


class BeliefAgent:
    def __init__(self, model_path, device="cpu"):
        self.model = bt.load_model(model_path, device)
        self.device = device
        self.static = None

    def start(self, header):
        self.static = bd.Static(header)

    @torch.no_grad()
    def beliefs(self, frame, seat):
        """Believed ships at every star, and the chance each of the seat's
        stars is attacked within the next cycle."""
        f = self.static.features(frame, seat)
        t = lambda x, dt=torch.float32: torch.as_tensor(x[None]).to(self.device, dt)
        batch = dict(
            star_x=bt.prepare_star_x(t(f["star_x"])),
            player_x=t(f["player_x"]),
            owner=t(f["star_owner_seat"], torch.long),
        )
        star_out, _ = self.model(batch)
        ships = torch.expm1(star_out[0, :, bt.ST["ships"]]).clamp(min=0)
        attack = torch.sigmoid(star_out[0, :, -1])
        return ships.cpu().numpy(), attack.cpu().numpy()

    def decide(self, observation, seat):
        ships, _ = self.beliefs(observation["frame"], seat)
        return {"beliefs": {"ships": [round(float(s), 1) for s in ships]}}


def placement(standings, seat):
    """1 = winner; then survivors by stars, then the defeated."""
    players = standings["players"]
    key = lambda p: (standings["winner"] != p["seat"], p["defeated"], -p["stars"])
    order = sorted(players, key=key)
    return [p["seat"] for p in order].index(seat) + 1


def play(model_path, mode="belief", seed=1, seat=9, max_ticks=3000, env=None):
    fog = mode != "omniscient"
    agent = BeliefAgent(model_path) if mode == "belief" else None
    own_env = env is None
    env = env or SolarisEnv()
    try:
        result = env.reset(agents=[seat], seed=seed, max_ticks=max_ticks, fog=fog)
        if agent:
            agent.start(env.header())
        last_cycle = -1
        while not result["done"]:
            for s in result["seats"]:
                obs = env.observe(s, frame=agent is not None)
                env.act(s, agent.decide(obs, s) if agent else {})
                if obs["cycle"] % 5 == 0 and obs["cycle"] != last_cycle:
                    last_cycle = obs["cycle"]
                    me = env.standings()["players"][s]
                    print(f"  tick {obs['tick']:>4}: {me['stars']} stars, {me['ships']} ships",
                          flush=True)
            result = env.step()
        standings = env.standings()
        return placement(standings, seat), standings
    finally:
        if own_env:
            env.close()


if __name__ == "__main__":
    model_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "belief"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seat = int(sys.argv[4]) if len(sys.argv) > 4 else 9
    max_ticks = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    place, standings = play(model_path, mode, seed, seat, max_ticks)
    me = standings["players"][seat]
    print(f"{mode}, seed {seed}, seat {seat}: placed {place} of {len(standings['players'])} "
          f"at tick {standings['tick']} with {me['stars']} stars")
