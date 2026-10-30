"""The commander policy: the network that makes an agent seat's decisions.

At each decision it sees what its seat sees (belief-model features from the
seat's frame) and the orders the built-in AI is considering, and decides

- a weight for each candidate order (defend, invade or claim a star), which
  multiplies the built-in AI's own score for it. Above 1 means "more
  urgent than the built-in AI thinks", below 1 "less";
and, at the start of each cycle,
- whether to switch research, and to what;
- a spending plan (SPENDING_PLANS): how the cycle's credits are split
  between science, industry, economy and saving;
- the seat's tuning (CONFIG_KNOBS), such as the split within the plan and
  how big a margin it wants before attacking, held for the cycle;
- whether to hire a star specialist with its specialist tokens, which, and
  at which of its stars.

The built-in executor carries the decision out.

It is built so that, before any training, it plays exactly like the
built-in AI: every head's last layer starts at zero, so order weights start
at 1, the tuning at the defaults, the spending plan at the built-in AI's,
specialists at "none" and research at "keep" (the built-in AI picks
research at random, which keeping leaves alone). Training then moves
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

# Spending plans: the share of credits upgraded at the start of each cycle
# (science, industry) and of what is left at its end (economy). The first
# is the built-in AI's; "save" keeps credits for carriers.
SPENDING_PLANS = [
    ("balanced", 20, 30, 100),
    ("economy", 10, 15, 100),
    ("industry", 10, 55, 100),
    ("science", 45, 20, 100),
    ("save", 10, 20, 30),
]

# Star specialists are chosen by id; index 0 is "hire none".
MAX_SPECIALIST_ID = 18

# name, default, how the network's output moves it, scale, lowest, highest.
# "add" adds scale * output; "mul" multiplies by exp(scale * output). The
# spending split's defaults come from the spending plan.
CONFIG_KNOBS = [
    ("firstTickSciencePercentage", 20, "add", 15.0, 0, 60),
    ("firstTickIndustryPercentage", 30, "add", 15.0, 0, 70),
    ("invasionAttackFactor", 1.5, "mul", 0.3, 1.05, 4.0),
    ("logisticStockpileCycles", 0.7, "mul", 0.3, 0.1, 3.0),
]

KEEP_BIAS = 4.0  # "keep research", the built-in plan and "hire none" start this far ahead
INITIAL_STD = 0.3


def knob_values(outputs, plan=0):
    """AiConfig values for the network's knob outputs, within a spending
    plan."""
    _, science, industry, economy = SPENDING_PLANS[plan]
    defaults = {"firstTickSciencePercentage": science, "firstTickIndustryPercentage": industry}
    config = {"lastTickEconomyPercentage": economy}
    for (name, default, how, scale, lo, hi), x in zip(CONFIG_KNOBS, outputs):
        default = defaults.get(name, default)
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
        self.plan_head = mlp(n_global, len(SPENDING_PLANS))
        # Which specialist (from the position and the tokens to spend), and
        # for each, how good each star is for it.
        self.specialist_head = mlp(n_global + 1, 1 + MAX_SPECIALIST_ID)
        self.specialist_star_head = mlp(d, MAX_SPECIALIST_ID)
        with torch.no_grad():
            for head in (self.research_head, self.plan_head, self.specialist_head):
                head[-1].bias[0] = KEEP_BIAS

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
        plan_logits = self.plan_head(g)

        # Specialists: only those affordable now, at stars the seat owns.
        budget = torch.log1p(b["spec_budget"]).unsqueeze(-1)
        spec_logits = self.specialist_head(torch.cat([g, budget], -1))
        spec_logits = spec_logits.masked_fill(~b["spec_mask"], -1e9)
        own = b["star_x"][..., bt.SF["owner_self"]] > 0
        star_logits = self.specialist_star_head(h)  # [B, S, specialist]
        star_logits = star_logits.masked_fill(~own.unsqueeze(-1), -1e9)
        return dict(order_mu=order_mu, config_mu=config_mu, research_logits=research_logits,
                    plan_logits=plan_logits, spec_logits=spec_logits, star_logits=star_logits)

    def value(self, b):
        """The critic's estimate of the outcome, from the true state."""
        _, _, h = self.critic_trunk.encode(b)
        return self.value_head(self._global(h, b)).squeeze(-1)

    def distributions(self, out, b, act=None):
        """The decision's distributions. The star distribution is for the
        specialist in act (or each decision's most likely one)."""
        spec = torch.distributions.Categorical(logits=out["spec_logits"])
        which = act["spec"] if act is not None else out["spec_logits"].argmax(-1)
        idx = (which - 1).clamp(min=0)
        star_logits = torch.gather(
            out["star_logits"], 2,
            idx.view(-1, 1, 1).expand(-1, out["star_logits"].shape[1], 1)).squeeze(-1)
        return dict(
            order=torch.distributions.Normal(out["order_mu"], self.order_log_std.exp()),
            config=torch.distributions.Normal(out["config_mu"], self.config_log_std.exp()),
            research=torch.distributions.Categorical(logits=out["research_logits"]),
            plan=torch.distributions.Categorical(logits=out["plan_logits"]),
            spec=spec,
            spec_star=torch.distributions.Categorical(logits=star_logits),
        )

    def log_prob(self, out, b, act):
        """Log-probability of the recorded actions, and entropy, per decision.
        Research, the spending plan, tuning and specialists are decided only
        at the start of a cycle, and count only then."""
        d = self.distributions(out, b, act)
        mask = b["cand_mask"].float()
        turn = b["research_turn"]
        hired = (act["spec"] > 0).float() * turn
        lp = ((d["order"].log_prob(act["order"]) * mask).sum(-1)
              + (d["config"].log_prob(act["config"]).sum(-1)
                 + d["research"].log_prob(act["research"])
                 + d["plan"].log_prob(act["plan"])
                 + d["spec"].log_prob(act["spec"])) * turn
              + d["spec_star"].log_prob(act["spec_star"]) * hired)
        ent = (self.order_log_std * mask.sum(-1)
               + (self.config_log_std.sum() + d["research"].entropy() + d["plan"].entropy()
                  + d["spec"].entropy()) * turn)
        return lp, ent


def batch_inputs(static, frame, seat, candidates, tick, research_turn, device="cpu",
                 affordable=(), budget=0):
    """Model inputs for one decision, as a batch of one. affordable lists
    the star specialists (ids) the seat can hire now, with budget tokens."""
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
    spec_mask = np.zeros(1 + MAX_SPECIALIST_ID, bool)
    spec_mask[0] = True
    if research_turn:
        spec_mask[[i for i in affordable if i <= MAX_SPECIALIST_ID]] = True
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
        spec_mask=t(spec_mask, torch.bool),
        spec_budget=t(float(budget)),
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
        b = batch_inputs(self.static, observation["frame"], seat, cands, tick, research_turn,
                         affordable=observation.get("affordableStarSpecialists", ()),
                         budget=observation.get("specialistBudget", 0))
        out = self.model(b)

        if self.sample:
            d = self.model.distributions(out, b)
            act = {k: d[k].sample() for k in ("order", "config", "research", "plan", "spec")}
            act["spec_star"] = self.model.distributions(out, b, act)["spec_star"].sample()
        else:
            act = dict(order=out["order_mu"], config=out["config_mu"],
                       research=out["research_logits"].argmax(-1),
                       plan=out["plan_logits"].argmax(-1), spec=out["spec_logits"].argmax(-1))
            act["spec_star"] = self.model.distributions(out, b, act)["spec_star"].logits.argmax(-1)
        logp, _ = self.model.log_prob(out, b, act)

        weights = torch.exp(act["order"][0]).tolist()
        decision = {"orderWeights": {c["key"]: w for c, w in zip(cands, weights)}}
        if research_turn:
            # Decided once a cycle and kept until the next.
            decision["config"] = knob_values(act["config"][0].tolist(), int(act["plan"][0]))
            r = int(act["research"][0])
            if r > 0:
                decision["research"] = self.static.techs[r - 1]
            spec = int(act["spec"][0])
            if spec > 0:
                decision["hire"] = [{"star": int(act["spec_star"][0]), "specialist": spec}]
        return decision, dict(inputs=b, act=act, logp=logp)
