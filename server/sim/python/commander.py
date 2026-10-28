"""The commander policy: the network that makes an agent seat's decisions.

At each decision it sees what its seat sees (belief-model features from the
seat's frame) and the orders the built-in AI is considering, and decides

- a weight for each candidate order (defend, invade or claim a star), which
  multiplies the built-in AI's own score for it. Above 1 means "more
  urgent than the built-in AI thinks", below 1 "less";
- the seat's tuning (CONFIG_KNOBS), such as the spending split and how big
  a margin it wants before attacking;
- at the start of each cycle, whether to switch research, and to what.

The built-in executor carries the decision out.

It is built so that, before any training, it plays exactly like the
built-in AI: every head's last layer starts at zero, so order weights start
at 1, the tuning at the defaults, and research at "keep" (the built-in AI
picks research at random, which keeping leaves alone). Training then moves
it away from there. Each decision can be sampled (for training, with
log-probabilities) or taken at its most likely value (for play).

The star encoding comes from a trained belief model, so the commander
starts out knowing what the fog hides. A separate critic reads the true
state (a truth frame), which the policy never sees.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belief_data as bd
import belief_train as bt

ORDER_KINDS = ["defend", "invade", "claim"]

# name, default, how the network's output moves it, scale, lowest, highest.
# "add" adds scale * output; "mul" multiplies by exp(scale * output).
CONFIG_KNOBS = [
    ("firstTickSciencePercentage", 20, "add", 15.0, 0, 60),
    ("firstTickIndustryPercentage", 30, "add", 15.0, 0, 70),
    ("invasionAttackFactor", 1.5, "mul", 0.3, 1.05, 4.0),
    ("logisticStockpileCycles", 0.7, "mul", 0.3, 0.1, 3.0),
]

KEEP_BIAS = 4.0  # "keep research" starts this far ahead of each switch
INITIAL_STD = 0.3


def knob_values(outputs):
    """AiConfig values for the network's knob outputs."""
    config = {}
    for (name, default, how, scale, lo, hi), x in zip(CONFIG_KNOBS, outputs):
        value = default + scale * x if how == "add" else default * math.exp(scale * x)
        value = min(max(value, lo), hi)
        config[name] = round(value) if how == "add" else value
    # The spending split cannot exceed the whole budget.
    over = config["firstTickSciencePercentage"] + config["firstTickIndustryPercentage"] - 100
    if over > 0:
        config["firstTickIndustryPercentage"] -= over
    return config


def zero_last(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), zero_last(nn.Linear(hidden, n_out)))


class Commander(nn.Module):
    def __init__(self, n_techs, d=128):
        super().__init__()
        n_star, n_player = len(bd.STAR_FEATURES), bt.N_STATS + n_techs
        self.n_techs = n_techs
        self.trunk = bt.BeliefModel(n_star, n_player, n_techs, d=d)
        self.critic_trunk = bt.BeliefModel(n_star, n_player, n_techs, d=d)

        # Candidate order: its star's encoding, its kind, the built-in AI's
        # score and how soon an attack lands.
        self.order_head = mlp(d + len(ORDER_KINDS) + 2, 1)
        self.order_log_std = nn.Parameter(torch.tensor(math.log(INITIAL_STD)))

        # The whole position: mean and max over own stars, own public stats,
        # and where in the cycle and game it is.
        n_global = 2 * d + n_player + 2
        self.config_head = mlp(n_global, len(CONFIG_KNOBS))
        self.config_log_std = nn.Parameter(torch.full((len(CONFIG_KNOBS),), math.log(INITIAL_STD)))
        self.research_head = mlp(n_global, 1 + n_techs)
        with torch.no_grad():
            self.research_head[-1].bias[0] = KEEP_BIAS

        self.value_head = nn.Sequential(
            nn.Linear(n_global, 128), nn.GELU(), nn.Linear(128, 1))

    def load_belief(self, path):
        """Start both trunks from a trained belief model."""
        saved = torch.load(path, map_location="cpu")
        self.trunk.load_state_dict(saved["model"])
        self.critic_trunk.load_state_dict(saved["model"])

    def _global(self, h, b):
        own = (b["star_x"][..., bt.SF["owner_self"]] > 0).unsqueeze(-1)
        n = own.sum(1).clamp(min=1)
        mean = (h * own).sum(1) / n
        mx = torch.where(own, h, torch.full_like(h, -1e4)).max(1).values
        mx = torch.where(own.any(1), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx, b["player_x"][:, 0], b["time"]], -1)

    def forward(self, b):
        """Distributions for every decision, for a batch of B decisions.

        b holds star_x [B,S,F], player_x [B,P,Fp], owner [B,S], time [B,2],
        cand_star [B,C], cand_kind [B,C], cand_score [B,C], cand_ticks [B,C]
        and cand_mask [B,C] (False for padding)."""
        _, _, h = self.trunk.encode(b)
        g = self._global(h, b)

        idx = b["cand_star"].clamp(min=0)
        ch = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        cx = torch.cat([
            ch,
            F.one_hot(b["cand_kind"], len(ORDER_KINDS)).float(),
            torch.log1p(b["cand_score"].clamp(min=0)).unsqueeze(-1),
            (b["cand_ticks"] / 20.0).unsqueeze(-1),
        ], -1)
        order_mu = self.order_head(cx).squeeze(-1)
        config_mu = self.config_head(g)
        research_logits = self.research_head(g)
        return dict(order_mu=order_mu, config_mu=config_mu, research_logits=research_logits)

    def value(self, b):
        """The critic's estimate of the outcome, from the true state."""
        _, _, h = self.critic_trunk.encode(b)
        return self.value_head(self._global(h, b)).squeeze(-1)

    def log_prob(self, out, b, act):
        """Log-probability of the recorded actions, and entropy, per decision."""
        mask = b["cand_mask"].float()
        std = self.order_log_std.exp()
        lp_orders = (torch.distributions.Normal(out["order_mu"], std)
                     .log_prob(act["order"]) * mask).sum(-1)
        cstd = self.config_log_std.exp()
        lp_config = torch.distributions.Normal(out["config_mu"], cstd).log_prob(act["config"]).sum(-1)
        research = torch.distributions.Categorical(logits=out["research_logits"])
        lp_research = research.log_prob(act["research"]) * b["research_turn"]
        lp = lp_orders + lp_config + lp_research
        ent = (self.order_log_std * mask.sum(-1) + self.config_log_std.sum()
               + research.entropy() * b["research_turn"])
        return lp, ent


def batch_inputs(static, frame, seat, candidates, tick, research_turn, device="cpu"):
    """Model inputs for one decision, as a batch of one."""
    f = static.features(frame, seat)
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)[None]).to(device, dt)
    C = max(len(candidates), 1)
    star = np.full(C, -1)
    kind = np.zeros(C, int)
    score = np.zeros(C, np.float32)
    ticks = np.zeros(C, np.float32)
    mask = np.zeros(C, bool)
    for i, c in enumerate(candidates):
        star[i] = c["starIndex"]
        kind[i] = ORDER_KINDS.index(c["kind"])
        score[i] = c["score"]
        ticks[i] = c.get("ticksUntil") or 0
        mask[i] = True
    cycle = static.cycle
    return dict(
        star_x=bt.prepare_star_x(t(f["star_x"])),
        player_x=t(f["player_x"]),
        owner=t(f["star_owner_seat"], torch.long),
        time=t([(tick % cycle) / cycle, tick / (cycle * 50)]),
        cand_star=t(star, torch.long),
        cand_kind=t(kind, torch.long),
        cand_score=t(score),
        cand_ticks=t(ticks),
        cand_mask=t(mask, torch.bool),
        research_turn=t(float(research_turn)),
    )


class CommanderAgent:
    """Plays seats with a Commander, through solaris_env.SolarisEnv."""

    def __init__(self, model, static, sample=False):
        self.model = model.eval()
        self.static = static
        self.sample = sample

    @torch.no_grad()
    def decide(self, observation, seat):
        """The decision to send for this observation, and what to record
        for training (inputs, sampled actions and their log-probability)."""
        tick = observation["tick"]
        research_turn = observation["ticksToProduction"] == self.static.cycle - 1
        cands = observation["candidates"]
        b = batch_inputs(self.static, observation["frame"], seat, cands, tick, research_turn)
        out = self.model(b)

        if self.sample:
            order = torch.normal(out["order_mu"], self.model.order_log_std.exp())
            config = torch.normal(out["config_mu"], self.model.config_log_std.exp())
            research = torch.distributions.Categorical(logits=out["research_logits"]).sample()
        else:
            order, config = out["order_mu"], out["config_mu"]
            research = out["research_logits"].argmax(-1)

        act = dict(order=order, config=config, research=research)
        logp, _ = self.model.log_prob(out, b, act)

        weights = torch.exp(order[0]).tolist()
        decision = {
            "orderWeights": {c["key"]: w for c, w in zip(cands, weights)},
            "config": knob_values(config[0].tolist()),
        }
        r = int(research[0])
        if research_turn and r > 0:
            decision["research"] = self.static.techs[r - 1]
        return decision, dict(inputs=b, act=act, logp=logp)
