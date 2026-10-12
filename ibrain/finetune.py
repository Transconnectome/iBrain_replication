"""M4: downstream velocity regression (SPEC 1.8, U8, U9, U11~U13, U27).
Three arms: finetune (full-parameter fine-tuning from a pretraining checkpoint) /
scratch (random init, same training) / ridge (linear regression on binned counts).
Labels are the mean behavior (velocity) per 100 ms patch of the window, (S, 2) (U9), z-scored per dimension
with train-set statistics before training (U27). Reading NWB files and attaching labels is in data_nwb.
M5 downstream: Brain Treebank binary classification scored by AUC (U35). The same model, heads and training loop,
with sig = IEEG so the iEEG encoder and type embedding of pretraining are used, one logit per window and
BCE loss. Reading the windows is in data_treebank."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ibrain.data_spike import P, S, collate, to_patches
from ibrain.data_treebank import TreebankWindows, collate_treebank
from ibrain.model import IEEG, SPIKE, IBrain
from ibrain.repro import load_checkpoint, seed_all


# ---------------- Labels and splits ----------------

def split_trials(windows, frac=0.8, seed=0):
    """U11: random 80/20 split by trial. A trial is (session_id, interval_id), because trial numbers repeat
    across sessions (Perich has one file per session). seed is fixed independently of the run seed so that
    the three arms share the same split (M4)."""
    def key(w):
        return w["session_id"], w["interval_id"]

    ids = sorted({key(w) for w in windows})
    np.random.default_rng(seed).shuffle(ids)
    train = set(ids[: int(round(frac * len(ids)))])
    return [w for w in windows if key(w) in train], [w for w in windows if key(w) not in train]


def synthetic_labeled(n_trials=200, n_units=30, seed=0):
    """M4 synthetic check: one 1 s window per trial. 2-D velocity = smooth random walk,
    unit firing rate = 0.5 softplus(v W + b) / bin, 20 ms Poisson counts. Velocity is an (almost) linear function
    of the firing rate, so ridge should reach a high R² and the model should also reach R² > 0."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (2, n_units))
    b = rng.normal(0, 0.5, n_units)
    out = []
    for i in range(n_trials):
        v = np.cumsum(rng.normal(0, 0.3, (S * P, 2)), 0)  # (50, 2)
        rate = 0.5 * np.log1p(np.exp(v @ W + b))
        out.append({"counts": rng.poisson(rate).astype(np.uint32), "vel": v.reshape(S, P, 2).mean(1),
                    "interval_id": str(i), "start_seconds": float(i), "session_id": "synthetic"})
    return out


class LabeledWindows(Dataset):
    """Window dict ("counts" [50, C], "vel" [S, 2]) -> ((C, S, P), (S, 2)). Labels become (vel - mu) / sd."""

    def __init__(self, windows, mu=0.0, sd=1.0):
        self.w = list(windows)
        self.mu, self.sd = mu, sd
        assert all(not np.isnan(w["vel"]).any() for w in self.w), "NaN label: must be filtered out by attach_velocity"

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        y = (np.asarray(self.w[i]["vel"], dtype=np.float64) - self.mu) / self.sd
        return to_patches(self.w[i]["counts"]), torch.as_tensor(y, dtype=torch.float32)


def collate_labeled(batch):
    x, valid = collate([b[0] for b in batch])
    return x, valid, torch.stack([b[1] for b in batch])


# ---------------- Task heads (U8) ----------------

class AttnPoolHead(nn.Module):
    """U8 default: per time patch, attention pooling over the channel axis (one learned query) -> Linear(d -> out).
    Padded channels are excluded from the softmax."""

    def __init__(self, d, C=None, out=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(d, d)
        self.out = nn.Linear(d, out)

    def forward(self, u, valid):  # (B, C, S, d), (B, C) -> (B, S, out)
        logit = (self.k(u) * self.q).sum(-1) / u.shape[-1] ** 0.5  # (B, C, S)
        a = logit.masked_fill(~valid[:, :, None], float("-inf")).softmax(1)
        return self.out((a[..., None] * u).sum(1))


class MeanPoolHead(nn.Module):
    """U8 alternative 1: per time patch, mean over valid channels -> Linear."""

    def __init__(self, d, C=None, out=2):
        super().__init__()
        self.out = nn.Linear(d, out)

    def forward(self, u, valid):
        w = valid[:, :, None, None].to(u.dtype)
        return self.out((u * w).sum(1) / w.sum(1))


class FlattenHead(nn.Module):
    """U8 alternative 2: flatten channels -> Linear(C*d -> out). Session-specific (fixed C). Padded channels are 0."""

    def __init__(self, d, C, out=2):
        super().__init__()
        self.out = nn.Linear(C * d, out)

    def forward(self, u, valid):
        B, C, S_, d = u.shape
        u = u * valid[:, :, None, None].to(u.dtype)
        return self.out(u.permute(0, 2, 1, 3).reshape(B, S_, C * d))


HEADS = {"attn": AttnPoolHead, "mean": MeanPoolHead, "flatten": FlattenHead}


class WindowLogit(nn.Module):
    """U35: one logit per window for classification. A U8 head with out=1 gives one logit per time patch (B, S, 1);
    the window logit is their mean over the 10 patches. -> (B,)"""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, u, valid):
        return self.head(u, valid)[..., 0].mean(1)


class Regressor(nn.Module):
    """Task head on top of the pretrained model's encode (Eq. 5). sig picks the encoder and type embedding, as in
    pretraining (model.encode): SPIKE for the M4 tasks, IEEG for Brain Treebank. The time embedding and the backbone
    are shared. Parameters of the other signal type get no gradient, so AdamW leaves them unchanged (U24).
    If frozen, the backbone runs with no-grad in eval mode (no dropout)."""

    def __init__(self, model, head, frozen=False, sig=SPIKE):
        super().__init__()
        self.model, self.head, self.frozen, self.sig = model, head, frozen, sig
        if frozen:
            model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x, valid):
        if self.frozen:
            with torch.no_grad():
                u = self.model.encode(x, valid, self.sig)
        else:
            u = self.model.encode(x, valid, self.sig)
        return self.head(u, valid)


# ---------------- Training and evaluation ----------------

def r2(pred, y):
    """Mean of the per-output-dimension R² (U27, sklearn r2_score default). pred, y (N, D)."""
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def train_regressor(reg, dl, epochs, lr=1e-4, wd=5e-2, device="cpu", loss_fn=F.mse_loss):
    """AdamW, clip 1.0, constant lr (U27). loss_fn: MSE for regression, BCE with logits for classification (U35).
    Returns the mean loss per epoch."""
    reg.to(device).train()
    opt = torch.optim.AdamW([p for p in reg.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    hist = []
    for _ in range(epochs):
        tot, n = 0.0, 0
        for x, valid, y in dl:
            x, valid, y = x.to(device), valid.to(device), y.to(device)
            loss = loss_fn(reg(x, valid), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        hist.append(tot / n)
    return hist


@torch.no_grad()
def evaluate(reg, dl, device="cpu"):
    reg.eval()
    preds, ys = [], []
    for x, valid, y in dl:
        preds.append(reg(x.to(device), valid.to(device)).cpu())
        ys.append(y)
    D = ys[0].shape[-1]
    return r2(torch.cat(preds).reshape(-1, D), torch.cat(ys).reshape(-1, D))


def fit_arm(train, test, model_cfg, seed, ckpt=None, head="attn", frozen=False, epochs=20, lr=1e-4, wd=5e-2,
            batch_size=32, device="cpu"):
    """One arm, one seed: finetune if ckpt is given, else scratch. Returns (test R², per-epoch loss history)."""
    seed_all(seed)
    model = IBrain(**model_cfg)
    if ckpt:
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
    C = max(w["counts"].shape[1] for w in train + test)
    reg = Regressor(model, HEADS[head](model_cfg.get("d", 256), C), frozen)
    # U27: z-score labels with train statistics. Raw velocities are tens to hundreds and the output layer cannot
    # reach that scale at lr 1e-4. R² per dimension is unchanged by the same affine map on prediction and target.
    Y = np.concatenate([np.asarray(w["vel"], dtype=np.float64) for w in train])
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(LabeledWindows(train, mu, sd), batch_size, shuffle=True, collate_fn=collate_labeled,
                       generator=g)
    dl_te = DataLoader(LabeledWindows(test, mu, sd), batch_size, collate_fn=collate_labeled)
    hist = train_regressor(reg, dl_tr, epochs, lr, wd, device)
    return evaluate(reg, dl_te, device), hist


def auc(score, y):
    """Area under the ROC curve (Mann-Whitney U), ties counted as one half. score, y (N,), y in {0, 1}."""
    from scipy.stats import rankdata
    score, y = np.asarray(score, dtype=np.float64), np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError(f"AUC needs both classes, got {n1} positives and {n0} negatives")
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ieeg_steps(ckpt):
    """Number of iEEG optimizer steps recorded in a pretraining checkpoint's history. 0 for a spike-only run: its
    iEEG encoder and iEEG type embedding are still at their random initialization."""
    return sum(h.get("sig") == IEEG for h in load_checkpoint(ckpt)["hist"])


@torch.no_grad()
def evaluate_auc(clf, dl, device="cpu"):
    clf.eval()
    logits, ys = [], []
    for x, valid, y in dl:
        logits.append(clf(x.to(device), valid.to(device)).cpu())
        ys.append(y)
    return auc(torch.cat(logits).numpy(), torch.cat(ys).numpy())


def fit_classifier_arm(train, test, model_cfg, seed, ckpt=None, head="attn", frozen=False, epochs=20, lr=1e-4,
                       wd=5e-2, batch_size=32, device="cpu"):
    """One arm, one seed of a Brain Treebank task (U35): finetune if ckpt is given, else scratch. train and test are
    data_treebank window dicts of one subject. The model encodes with sig = IEEG; the U8 head gives one logit per
    window (WindowLogit); BCE with logits; the rest as U27. Labels are 0/1, so no z-scoring.
    Returns (test AUC, per-epoch loss history)."""
    seed_all(seed)
    model = IBrain(**model_cfg)
    if ckpt:
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
    C = max(w["wave"].shape[0] for w in train + test)
    clf = Regressor(model, WindowLogit(HEADS[head](model_cfg.get("d", 256), C, out=1)), frozen, sig=IEEG)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(TreebankWindows(train), batch_size, shuffle=True, collate_fn=collate_treebank, generator=g)
    dl_te = DataLoader(TreebankWindows(test), batch_size, collate_fn=collate_treebank)
    hist = train_regressor(clf, dl_tr, epochs, lr, wd, device, loss_fn=F.binary_cross_entropy_with_logits)
    return evaluate_auc(clf, dl_te, device), hist


def ridge_r2(train, test, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), seed=0):
    """U13 baseline: binned counts of the patch (P*C) -> velocity, closed-form ridge, ONE DECODER PER SESSION: units are
    different neurons in every session, so stacking them into shared feature columns would pool unrelated neurons (or
    fail when unit counts differ). Features are standardized per session. One λ for all sessions, chosen on an inner
    validation that re-splits train 80/20 by trial, then every session is refit on all of its train windows (U27).
    R² is over all test patches pooled across sessions, as for the neural arms."""
    def xy(ws):
        X = np.concatenate([np.asarray(w["counts"], np.float64).reshape(S, -1) for w in ws])  # (N*S, P*C)
        Y = np.concatenate([np.asarray(w["vel"], np.float64) for w in ws])  # (N*S, D)
        return X, Y

    def fit(X, Y, lam):
        mu, sd, ym = X.mean(0), X.std(0) + 1e-8, Y.mean(0)
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (Y - ym))
        return lambda Z: ((Z - mu) / sd) @ W + ym

    def by_session(ws):
        out = {}
        for w in ws:
            out.setdefault(w["session_id"], []).append(w)
        return out

    def pooled(tr, te, lam):
        """Fit each session on its tr windows, predict its te windows; (pred, target) pooled over sessions."""
        trs, preds, ys = by_session(tr), [], []
        for sess, ws in by_session(te).items():
            if sess not in trs:
                raise ValueError(f"session {sess!r} has held-out windows but no training windows")
            Xt, Yt = xy(ws)
            preds.append(fit(*xy(trs[sess]), lam)(Xt))
            ys.append(Yt)
        return np.concatenate(preds), np.concatenate(ys)

    inner_tr, inner_va = split_trials(train, 0.8, seed)
    lam = max(lams, key=lambda l: r2(*pooled(inner_tr, inner_va, l)))
    return r2(*pooled(train, test, lam)), lam
