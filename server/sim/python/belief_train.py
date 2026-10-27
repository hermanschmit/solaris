"""Train a first belief model on samples from belief_data.py. Needs PyTorch.

    python3 belief_data.py data/belief belief.npz
    python3 belief_train.py belief.npz [epochs] [out.pt]

The model reads one player's view of the galaxy (per star: what it sees or
remembers, plus the public stats and tech of the star's owner) and predicts
what fog hides:

- per star: true ships and infrastructure, true enemy threat within one and
  two cycles, and whether enemy ships arrive in the next cycle and how many;
- per enemy: credits and the technology being researched.

Stars attend to each other with a learned bias by distance, so a star's
estimate depends on what is near it in travel distance. Nothing about the
frontier is hard-coded beyond the features; the model learns how threat
falls off behind it.

Games are split into training and validation by seed, so validation games
are never seen in training. After each epoch it prints validation scores
next to simple baselines that use only what the player knows.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import belief_data as bd

SF = {f: i for i, f in enumerate(bd.STAR_FEATURES)}
ST = {f: i for i, f in enumerate(bd.STAR_TARGETS)}
N_STATS = len(bd.PLAYER_FEATURES)

# Distance buckets for the attention bias, in galaxy units. A carrier flies
# 300 units a cycle in standard games, and hyperspace range starts at 125.
BUCKET = 25.0
N_BUCKETS = 48  # the last bucket is everything beyond 1175


def log_ships(x):
    """Ship targets span 0 to thousands; train on log scale."""
    return torch.log1p(x * bd.SCALE_SHIPS)


class Data:
    def __init__(self, path, device):
        z = np.load(path)
        self.n = len(z["meta"])
        self.seeds = z["meta"][:, 0]
        t = lambda k, dt=torch.float32: torch.as_tensor(z[k]).to(dt)
        self.star_x = t("star_x")
        # "Never seen" is coded as age 1000; squash ages to a small range.
        age = self.star_x[..., SF["age"]]
        self.star_x[..., SF["age"]] = age.clamp(max=50) / 10
        self.star_y = t("star_y")
        self.known = t("star_known", torch.bool)
        self.owner = t("star_owner_seat", torch.long)
        self.player_x = t("player_x")
        self.player_y = t("player_y", torch.long)
        self.device = device

    def split(self, frac=0.1, seed=0):
        seeds = np.unique(self.seeds)
        rng = np.random.default_rng(seed)
        val = set(rng.choice(seeds, max(1, int(len(seeds) * frac)), replace=False))
        is_val = np.array([s in val for s in self.seeds])
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

    def batch(self, idx):
        d = self.device
        return dict(
            star_x=self.star_x[idx].to(d),
            star_y=self.star_y[idx].to(d),
            known=self.known[idx].to(d),
            owner=self.owner[idx].to(d),
            player_x=self.player_x[idx].to(d),
            player_y=self.player_y[idx].to(d),
        )


class Layer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dist_bias = nn.Embedding(N_BUCKETS, heads)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, buckets):
        B, S, D = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(B, S, 3, self.heads, -1).unbind(2)
        bias = self.dist_bias(buckets).permute(0, 3, 1, 2)  # B, heads, S, S
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
        h = h + self.out(a.transpose(1, 2).reshape(B, S, D))
        return h + self.ff(self.norm2(h))


class BeliefModel(nn.Module):
    def __init__(self, star_features, player_features, n_techs, d=128, layers=4, heads=8):
        super().__init__()
        self.n_techs = n_techs
        # Star features, plus the owner's public stats and tech, plus the
        # observer's own.
        self.embed = nn.Linear(star_features + 2 * player_features, d)
        self.layers = nn.ModuleList(Layer(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.star_head = nn.Linear(d, len(bd.STAR_TARGETS) + 1)  # + attack logit
        nn.init.zeros_(self.star_head.weight)
        nn.init.zeros_(self.star_head.bias)
        self.player_head = nn.Sequential(
            nn.Linear(d + player_features, d), nn.GELU(), nn.Linear(d, 1 + n_techs))

    def forward(self, b):
        sx, px, owner = b["star_x"], b["player_x"], b["owner"]
        B, S, _ = sx.shape
        P = px.shape[1]

        owned = owner >= 0
        owner_px = torch.gather(
            px, 1, owner.clamp(min=0).unsqueeze(-1).expand(-1, -1, px.shape[-1]))
        owner_px = owner_px * owned.unsqueeze(-1)
        self_px = px[:, :1].expand(-1, S, -1)

        xy = sx[..., [SF["x"], SF["y"]]] * 100.0
        buckets = (torch.cdist(xy, xy) / BUCKET).long().clamp(max=N_BUCKETS - 1)

        h = self.embed(torch.cat([sx, owner_px, self_px], -1))
        for layer in self.layers:
            h = layer(h, buckets)
        h = self.norm(h)

        # Predict corrections to what the player already knows: the last-seen
        # ships and infrastructure, the known threat, and the ships seen
        # heading in. The untrained model starts at those baselines.
        known = torch.stack([
            torch.log1p(sx[..., SF["ships"]] * bd.SCALE_SHIPS),
            *(torch.log1p(sx[..., SF[k]] * bd.SCALE_INFRA) for k in ("economy", "industry", "science")),
            torch.log1p(sx[..., SF["known_threat_1c"]] * bd.SCALE_SHIPS),
            torch.log1p(sx[..., SF["known_threat_2c"]] * bd.SCALE_SHIPS),
            torch.log1p(sx[..., SF["seen_carriers_in"]] * bd.SCALE_SHIPS),
        ], -1)
        star_out = self.star_head(h)
        star_out = torch.cat([star_out[..., :-1] + known, star_out[..., -1:]], -1)

        # Each player's summary: the mean over the stars it owns.
        one_hot = F.one_hot(owner.clamp(min=0), P).float() * owned.unsqueeze(-1)
        pooled = torch.einsum("bsp,bsd->bpd", one_hot, h) / one_hot.sum(1).clamp(min=1).unsqueeze(-1)
        player_out = self.player_head(torch.cat([pooled, px], -1))

        return star_out, player_out


def losses(model, b):
    star_out, player_out = model(b)
    y = b["star_y"]
    own = b["star_x"][..., SF["owner_self"]] > 0
    enemy = b["star_x"][..., SF["owner_enemy"]] > 0
    hidden = enemy & ~b["known"]

    def masked(pred, target, mask):
        return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)

    # Ship counts are scored on their own scale (hundreds of ships), with
    # a Huber loss, so the model is judged the way the error is reported.
    def ships(k, mask):
        pred = torch.expm1(star_out[..., ST[k]].clamp(max=12)) / bd.SCALE_SHIPS
        err = F.smooth_l1_loss(pred, y[..., ST[k]], reduction="none", beta=0.1)
        return (err * mask).sum() / mask.sum().clamp(min=1)

    L = {}
    L["ships"] = ships("ships", hidden)
    infra = torch.log1p(y[..., 1:4] * bd.SCALE_INFRA)
    L["infra"] = masked(star_out[..., 1:4], infra, hidden.unsqueeze(-1).float()) / 3
    L["threat"] = (ships("threat_1c", own) + ships("threat_2c", own)) / 2
    L["incoming"] = ships("incoming_1c", own)
    attacked = (y[..., ST["incoming_1c"]] > 0).float()
    L["attack"] = (F.binary_cross_entropy_with_logits(
        star_out[..., -1], attacked, reduction="none") * own).sum() / own.sum().clamp(min=1)

    # Enemies only (seat 0 is the observer), and only those still playing.
    alive = b["player_x"][..., 1] == 0
    alive[:, 0] = False
    credits = torch.log1p(b["player_y"][..., 0].float().clamp(min=0))
    L["credits"] = masked(player_out[..., 0], credits, alive)
    research = b["player_y"][..., 2]
    ok = alive & (research >= 0)
    if ok.any():
        L["research"] = F.cross_entropy(player_out[..., 1:][ok], research[ok])
    return L, star_out, player_out


@torch.no_grad()
def evaluate(model, data, idx, batch_size):
    model.eval()
    agg = {}

    def add(k, v, n=1):
        s, c = agg.get(k, (0.0, 0))
        agg[k] = (s + float(v), c + n)

    probs, labels, base = [], [], []
    for i in range(0, len(idx), batch_size):
        b = data.batch(idx[i:i + batch_size])
        L, star_out, player_out = losses(model, b)
        for k, v in L.items():
            add("loss_" + k, v)
        y, x = b["star_y"], b["star_x"]
        own = x[..., SF["owner_self"]] > 0
        enemy = x[..., SF["owner_enemy"]] > 0
        hidden = enemy & ~b["known"]

        true_ships = y[..., ST["ships"]][hidden] * bd.SCALE_SHIPS
        pred_ships = torch.expm1(star_out[..., 0][hidden]).clamp(min=0)
        add("ships_mae", (pred_ships - true_ships).abs().sum(), hidden.sum())
        add("ships_mae_zero", true_ships.abs().sum(), hidden.sum())

        t1 = y[..., ST["threat_1c"]][own] * bd.SCALE_SHIPS
        p1 = torch.expm1(star_out[..., ST["threat_1c"]][own]).clamp(min=0)
        k1 = x[..., SF["known_threat_1c"]][own] * bd.SCALE_SHIPS
        add("threat_mae", (p1 - t1).abs().sum(), own.sum())
        add("threat_mae_known", (k1 - t1).abs().sum(), own.sum())

        probs.append(torch.sigmoid(star_out[..., -1][own]).cpu())
        labels.append((y[..., ST["incoming_1c"]][own] > 0).cpu())
        base.append(k1.cpu())

        alive = b["player_x"][..., 1] == 0
        alive[:, 0] = False
        r = b["player_y"][..., 2]
        ok = alive & (r >= 0)
        pred = player_out[..., 1:].argmax(-1)
        add("research_acc", (pred[ok] == r[ok]).sum(), ok.sum())

    out = {k: s / max(c, 1) for k, (s, c) in agg.items()}
    p, l, kb = torch.cat(probs), torch.cat(labels), torch.cat(base)
    out["attack_auc"] = auc(p, l)
    out["attack_auc_known_threat"] = auc(kb, l)
    out["attack_rate"] = l.float().mean().item()
    model.train()
    return out


def auc(score, label):
    """Chance that a random attacked star scores above a random safe one."""
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    pos = label.sum().item()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return ((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)).item()


def main():
    path = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out_path = sys.argv[3] if len(sys.argv) > 3 else "belief_model.pt"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    data = Data(path, device)
    train_idx, val_idx = data.split()
    n_techs = data.player_x.shape[-1] - N_STATS
    model = BeliefModel(data.star_x.shape[-1], data.player_x.shape[-1], n_techs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps = epochs * (len(train_idx) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=max(steps, 1))

    print(f"{data.n} samples ({len(train_idx)} train, {len(val_idx)} validation, "
          f"split by game), {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"parameters, on {device}")

    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        t = time.time()
        perm = rng.permutation(train_idx)
        total = 0.0
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            L, _, _ = losses(model, data.batch(perm[i:i + batch_size]))
            loss = sum(L.values())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        v = evaluate(model, data, val_idx, batch_size)
        print(f"epoch {epoch + 1:>3}  train loss {total / max(len(perm) // batch_size, 1):.3f}  "
              f"[{time.time() - t:.0f}s]\n"
              f"   attacked next cycle (own stars, {v['attack_rate']:.0%} are): "
              f"AUC {v['attack_auc']:.3f} vs {v['attack_auc_known_threat']:.3f} from known threat\n"
              f"   threat within a cycle, mean error in ships: "
              f"{v['threat_mae']:.1f} vs {v['threat_mae_known']:.1f} from known threat\n"
              f"   ships at hidden enemy stars, mean error: "
              f"{v['ships_mae']:.1f} vs {v['ships_mae_zero']:.1f} guessing zero\n"
              f"   enemy research, accuracy: {v['research_acc']:.1%} "
              f"(chance {1 / n_techs:.0%})")

    torch.save({"model": model.state_dict(), "star_features": bd.STAR_FEATURES,
                "n_techs": n_techs}, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
