import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
import config                                                   # all paths live in config.py
os.chdir(config.WORK_DIR)                                       # outputs (.npz/.png/checkpoints) land in outputs/

# ============================================================
#  Train the conditional normalizing flow (CNF) for Bayesian 1D TEM inversion
#  (neural posterior estimation: the flow learns the full posterior p(m | d)).
#
#  Architecture
#    context encoder : two-branch ResNet (LM and HM branches) -> 128-dim context
#    flow            : zuko Neural Spline Flow with one feature per layer
#                      (log10 rho, standardised), conditioned on the context
#
#  Objective: maximise log p(m | d) over (m, d) pairs
#     loss = -flow(context(d)).log_prob(m)
#
#  Noise: ~3 % multiplicative log-Gaussian noise is added to d during
#  training, matching the ~3 % error bars of the field data. Without it the
#  learned posterior would collapse towards a delta function.
#
#  LM late-time gate truncation: the last LM_GATES_TO_DROP gates of the LM
#  curve can be dropped before normalisation (they sit near the noise floor
#  for many soundings). The encoder does not depend on sequence length, so
#  only N_LM changes. The count is saved in norm_stats.npz and checked by
#  the evaluation scripts; field data must have the same trailing gates
#  removed, by index.
#
#  DOI: the generator stores a per-sample depth of investigation (DOI). It is
#  not used in the loss. It serves afterwards as an independent check that the
#  learned posterior width grows around the physical DOI.
#
#  Usage after training:
#     posterior = flow(context(d_field))
#     samples   = posterior.sample((5000,))
#
#  Requires:  pip install zuko
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time, json, math
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import zuko
except ImportError:
    raise ImportError("Install zuko first:  pip install zuko")

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = str(config.TRAIN_NPZ)    # written by 01_generate_training_data.py

OUT_DIR     = Path("nn_cnf_1"); OUT_DIR.mkdir(exist_ok=True)

VAL_FRAC, TEST_FRAC = 0.10, 0.05
BATCH_SIZE   = 256
EPOCHS       = 100
LR_MAX       = 5e-4
LR_MIN       = 1e-6
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-5
NOISE_STD_LOG = 0.013     # ~3% multiplicative noise: log10(1.03) ≈ 0.013
                          # matches the ERROR_BAR=3.0 (%) in the .usf file
CONTEXT_DIM  = 128
FLOW_TRANSFORMS = 5       # number of NSF coupling layers
FLOW_HIDDEN  = [256, 256] # hyper-network size inside each transform
FLOW_BINS    = 8          # spline bins
PATIENCE     = 30
SEED         = 42

# ---- LM late-time gate truncation ----
LM_GATES_TO_DROP = 0      # drop the last N LM gates (noisy late-time LM)
                          # set to 0 to disable truncation entirely

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================
d = np.load(DATA_PATH)
M      = d["m_log10_rho"]
DLM_full = d["d_lm"]
DHM      = d["d_hm"]

N_LM_FULL = DLM_full.shape[1]
if LM_GATES_TO_DROP > 0:
    if LM_GATES_TO_DROP >= N_LM_FULL:
        raise ValueError(
            f"LM_GATES_TO_DROP={LM_GATES_TO_DROP} >= total LM gates "
            f"({N_LM_FULL}); nothing would be left."
        )
    DLM = DLM_full[:, :-LM_GATES_TO_DROP]
else:
    DLM = DLM_full

N, N_LAY   = M.shape
N_LM, N_HM = DLM.shape[1], DHM.shape[1]
print(f"Loaded {N} samples,  LM {N_LM} (dropped last {LM_GATES_TO_DROP} of "
      f"{N_LM_FULL}),  HM {N_HM},  layers {N_LAY}")

# doi/layer_weight are diagnostic-only (see note above) — load if present,
# don't fail if an older dataset without them is used.
DOI_ALL = d["doi"] if "doi" in d.files else None
ZC      = d["zc"]  if "zc"  in d.files else None

# If the dataset stores LM gate times, truncate them too so norm_stats.npz
# can carry the retained gate times through for the inference script.
LM_GATE_TIMES_FULL = d["t_lm"] if "t_lm" in d.files else None
LM_GATE_TIMES = (LM_GATE_TIMES_FULL[:-LM_GATES_TO_DROP]
                  if (LM_GATE_TIMES_FULL is not None and LM_GATES_TO_DROP > 0)
                  else LM_GATE_TIMES_FULL)

X_lm_raw = np.log10(DLM).astype(np.float32)
X_hm_raw = np.log10(DHM).astype(np.float32)
Y_raw    = M.astype(np.float32)

rng = np.random.default_rng(SEED)
idx = rng.permutation(N)
n_test = int(TEST_FRAC * N); n_val = int(VAL_FRAC * N)
i_test = idx[:n_test]; i_val = idx[n_test:n_test+n_val]; i_tr = idx[n_test+n_val:]

xlm_mean = X_lm_raw[i_tr].mean(0); xlm_std = X_lm_raw[i_tr].std(0) + 1e-8
xhm_mean = X_hm_raw[i_tr].mean(0); xhm_std = X_hm_raw[i_tr].std(0) + 1e-8
y_mean   = Y_raw   [i_tr].mean(0); y_std   = Y_raw   [i_tr].std(0) + 1e-8

def norm_lm(a): return (a - xlm_mean) / xlm_std
def norm_hm(a): return (a - xhm_mean) / xhm_std
def norm_y (a): return (a - y_mean)   / y_std
def unnorm_y(a): return a * y_std + y_mean

Xlm_tr, Xhm_tr, Y_tr = norm_lm(X_lm_raw[i_tr]),  norm_hm(X_hm_raw[i_tr]),  norm_y(Y_raw[i_tr])
Xlm_va, Xhm_va, Y_va = norm_lm(X_lm_raw[i_val]), norm_hm(X_hm_raw[i_val]), norm_y(Y_raw[i_val])
Xlm_te, Xhm_te, Y_te = norm_lm(X_lm_raw[i_test]),norm_hm(X_hm_raw[i_test]),norm_y(Y_raw[i_test])

# Save norm stats PLUS the truncation config, so any inference script can
# (a) apply the identical normalization, and (b) assert the field data was
# truncated the same way before it silently produces garbage.
norm_stats_payload = dict(
    xlm_mean=xlm_mean, xlm_std=xlm_std,
    xhm_mean=xhm_mean, xhm_std=xhm_std,
    y_mean=y_mean, y_std=y_std,
    lm_gates_to_drop=np.array(LM_GATES_TO_DROP),
    n_lm_full=np.array(N_LM_FULL),
    n_lm_used=np.array(N_LM),
)
if LM_GATE_TIMES is not None:
    norm_stats_payload["lm_gate_times_used"] = LM_GATE_TIMES
np.savez(OUT_DIR / "norm_stats.npz", **norm_stats_payload)

def as_ds(xlm, xhm, y):
    return TensorDataset(
        torch.from_numpy(xlm).unsqueeze(1),
        torch.from_numpy(xhm).unsqueeze(1),
        torch.from_numpy(y))

dl_tr = DataLoader(as_ds(Xlm_tr, Xhm_tr, Y_tr), batch_size=BATCH_SIZE,
                   shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
dl_va = DataLoader(as_ds(Xlm_va, Xhm_va, Y_va), batch_size=BATCH_SIZE, shuffle=False)
dl_te = DataLoader(as_ds(Xlm_te, Xhm_te, Y_te), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# CONTEXT ENCODER  — two ResNet branches (LM, HM), regression head removed
#   (unchanged: AdaptiveAvgPool1d makes this sequence-length agnostic,
#    so LM truncation needs no architecture edits here)
# ============================================================
class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel=3, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.act   = nn.LeakyReLU(0.1, inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.drop(self.act(x + r))

class Branch(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True))
        self.res1 = ResBlock1D(32, 3, dropout)
        self.res2 = ResBlock1D(32, 3, dropout)
        self.expand = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True))
        self.res3 = ResBlock1D(64, 3, dropout)
        self.res4 = ResBlock1D(64, 3, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x); x = self.res2(x)
        x = self.expand(x)
        x = self.res3(x); x = self.res4(x)
        return self.pool(x).squeeze(-1)          # (B, 64)

class ContextEncoder(nn.Module):
    """(xlm, xhm) -> context vector for the flow."""
    def __init__(self, context_dim=128, dropout=0.1):
        super().__init__()
        self.lm_branch = Branch(dropout)
        self.hm_branch = Branch(dropout)
        self.proj = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout),
            nn.Linear(256, context_dim))
    def forward(self, xlm, xhm):
        f = torch.cat([self.lm_branch(xlm), self.hm_branch(xhm)], 1)
        return self.proj(f)                       # (B, context_dim)

# ============================================================
# THE CONDITIONAL FLOW
# ============================================================
encoder = ContextEncoder(CONTEXT_DIM).to(DEVICE)
flow = zuko.flows.NSF(
    features=N_LAY,
    context=CONTEXT_DIM,
    transforms=FLOW_TRANSFORMS,
    bins=FLOW_BINS,
    hidden_features=FLOW_HIDDEN,
).to(DEVICE)

n_params_enc  = sum(p.numel() for p in encoder.parameters())
n_params_flow = sum(p.numel() for p in flow.parameters())
print(f"Encoder params: {n_params_enc:,}   Flow params: {n_params_flow:,}   "
      f"Total: {n_params_enc + n_params_flow:,}")

params = list(encoder.parameters()) + list(flow.parameters())
opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY)

def get_lr(epoch):
    if epoch <= WARMUP_EPOCHS:
        return LR_MAX * epoch / WARMUP_EPOCHS
    prog = min((epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS), 1.0)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * prog))

# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================
xlm_std_t = torch.from_numpy(xlm_std).to(DEVICE)
xhm_std_t = torch.from_numpy(xhm_std).to(DEVICE)

def add_noise(xlm, xhm):
    """~3% multiplicative noise in linear space == 0.013 in log10, applied
    on the standardized inputs (rescaled by per-gate std)."""
    xlm = xlm + torch.randn_like(xlm) * (NOISE_STD_LOG / xlm_std_t.view(1, 1, -1))
    xhm = xhm + torch.randn_like(xhm) * (NOISE_STD_LOG / xhm_std_t.view(1, 1, -1))
    return xlm, xhm

def evaluate_nll(dl):
    encoder.eval(); flow.eval()
    tot = 0.0; n = 0
    with torch.no_grad():
        for xlm, xhm, y in dl:
            xlm, xhm, y = xlm.to(DEVICE), xhm.to(DEVICE), y.to(DEVICE)
            xlm, xhm = add_noise(xlm, xhm)
            ctx = encoder(xlm, xhm)
            l = -flow(ctx).log_prob(y).mean()
            tot += float(l) * xlm.shape[0]; n += xlm.shape[0]
    return tot / n

# ============================================================
# PERIODIC POSTERIOR-MEAN R2  (every R2_EVERY epochs, small subset)
# ============================================================
R2_EVERY  = 5
R2_N_VAL  = 500
R2_N_SAMP = 100
r2_sel = np.random.default_rng(1).choice(len(Y_va),
                                         size=min(R2_N_VAL, len(Y_va)),
                                         replace=False)

def quick_posterior_r2():
    encoder.eval(); flow.eval()
    with torch.no_grad():
        xlm = torch.from_numpy(Xlm_va[r2_sel][:, None, :]).to(DEVICE)
        xhm = torch.from_numpy(Xhm_va[r2_sel][:, None, :]).to(DEVICE)
        xlm, xhm = add_noise(xlm, xhm)
        ctx  = encoder(xlm, xhm)                    # (500, ctx)
        samp = flow(ctx).sample((R2_N_SAMP,))       # (100, 500, 20)
        pm   = samp.mean(0).cpu().numpy()           # posterior mean
    P = unnorm_y(pm); Y = unnorm_y(Y_va[r2_sel])
    ss_res = ((Y - P) ** 2).sum()
    ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot)

# ============================================================
# TRAINING LOOP
# ============================================================
history = {"epoch": [], "lr": [], "train_nll": [], "val_nll": [],
           "val_r2_epochs": [], "val_r2": []}
best_val = np.inf; best_epoch = -1; wait = 0
best_enc = None; best_flow = None
t0 = time.time()

print(f"\n{'Ep':>4} {'lr':>9} {'trNLL':>9} {'vaNLL':>9} {'best':>5} {'t/ep(s)':>8}")
print("-" * 52)

for epoch in range(1, EPOCHS + 1):
    lr_now = get_lr(epoch)
    for g in opt.param_groups: g["lr"] = lr_now

    ep_t0 = time.time()
    encoder.train(); flow.train()
    ep_loss = 0.0; n_seen = 0

    for xlm, xhm, y in dl_tr:
        xlm, xhm, y = xlm.to(DEVICE), xhm.to(DEVICE), y.to(DEVICE)
        xlm, xhm = add_noise(xlm, xhm)
        opt.zero_grad()
        ctx = encoder(xlm, xhm)
        loss = -flow(ctx).log_prob(y).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        ep_loss += float(loss) * xlm.shape[0]; n_seen += xlm.shape[0]

    tr_nll = ep_loss / n_seen
    va_nll = evaluate_nll(dl_va)

    improved = va_nll < best_val - 1e-5
    if improved:
        best_val = va_nll; best_epoch = epoch; wait = 0
        best_enc  = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        best_flow = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
    else:
        wait += 1

    history["epoch"].append(epoch); history["lr"].append(lr_now)
    history["train_nll"].append(tr_nll); history["val_nll"].append(va_nll)

    r2_str = ""
    if epoch % R2_EVERY == 0 or epoch == 1:
        va_r2 = quick_posterior_r2()
        history["val_r2_epochs"].append(epoch)
        history["val_r2"].append(va_r2)
        r2_str = f"  vaR2={va_r2:.4f}"
        encoder.train(); flow.train()

    print(f"{epoch:4d} {lr_now:9.2e} {tr_nll:9.4f} {va_nll:9.4f} "
          f"{'*' if improved else ' ':>5} {time.time()-ep_t0:8.2f}{r2_str}",
          flush=True)

    if wait >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (best {best_epoch})")
        break

print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")
encoder.load_state_dict(best_enc); flow.load_state_dict(best_flow)
torch.save(encoder.state_dict(), OUT_DIR / "encoder_best.pt")
torch.save(flow.state_dict(),    OUT_DIR / "flow_best.pt")
with open(OUT_DIR / "history.json", "w") as f:
    json.dump(history, f)

# ============================================================
# POSTERIOR EVALUATION ON TEST SET
# ============================================================
N_POST_SAMPLES = 500
N_TEST_EVAL    = 1000

encoder.eval(); flow.eval()

sel = np.random.default_rng(0).choice(len(Y_te), size=min(N_TEST_EVAL, len(Y_te)),
                                      replace=False)
post_mean = np.zeros((len(sel), N_LAY), dtype=np.float32)
post_std  = np.zeros((len(sel), N_LAY), dtype=np.float32)
ranks     = np.zeros((len(sel), N_LAY), dtype=np.int32)

with torch.no_grad():
    for k, i in enumerate(sel):
        xlm = torch.from_numpy(Xlm_te[i][None, None, :]).to(DEVICE)
        xhm = torch.from_numpy(Xhm_te[i][None, None, :]).to(DEVICE)
        xlm, xhm = add_noise(xlm, xhm)
        ctx = encoder(xlm, xhm)
        samp = flow(ctx).sample((N_POST_SAMPLES,)).squeeze(1).cpu().numpy()
        post_mean[k] = samp.mean(0)
        post_std[k]  = samp.std(0)
        ranks[k] = (samp < Y_te[i][None, :]).sum(0)

P_un = unnorm_y(post_mean); Ytrue_un = unnorm_y(Y_te[sel])
ss_res = ((Ytrue_un - P_un) ** 2).sum()
ss_tot = ((Ytrue_un - Ytrue_un.mean(0, keepdims=True)) ** 2).sum()
r2_mean = 1 - ss_res / ss_tot
print(f"\nPosterior-mean R^2 on {len(sel)} test points: {r2_mean:.4f}")
print("(Compare against ResNet point estimate. Similar or slightly lower")
print(" is normal — the flow trades a bit of point accuracy for the")
print(" full distribution.)")

u = (ranks + 0.5) / N_POST_SAMPLES

def coverage(u, level):
    lo = (1 - level) / 2; hi = 1 - lo
    return ((u > lo) & (u < hi)).mean(axis=0)

cov68 = coverage(u, 0.68)
cov95 = coverage(u, 0.95)
print("\nCALIBRATION (per-layer empirical coverage):")
print(f"{'layer':>5} {'68% CI cov':>11} {'95% CI cov':>11}")
for i in range(N_LAY):
    flag68 = "" if 0.62 < cov68[i] < 0.74 else "  <-- off"
    print(f"{i+1:5d} {cov68[i]:11.3f} {cov95[i]:11.3f}{flag68}")
print(f"\nMean coverage:  68% CI -> {cov68.mean():.3f}   "
      f"95% CI -> {cov95.mean():.3f}")
print("(Well-calibrated: 68% coverage ≈ 0.68, 95% ≈ 0.95.")
print(" Under-coverage = overconfident; over-coverage = underconfident.)")

# ---- DOI sanity check (diagnostic only, not used in training) ----
doi_te_sel = None
if DOI_ALL is not None and ZC is not None:
    doi_te_sel = DOI_ALL[i_test][sel]
    print(f"\nDOI over these {len(sel)} test points: mean {doi_te_sel.mean():.1f} m, "
          f"median {np.median(doi_te_sel):.1f} m")
    # crude check: does posterior std trend upward once past each sample's DOI?
    zc_chk = ZC
    below_doi = zc_chk[None, :] > doi_te_sel[:, None]
    std_above = post_std[~below_doi].mean() * y_std.mean()
    std_below = post_std[below_doi].mean()  * y_std.mean() if below_doi.any() else np.nan
    print(f"Mean learned posterior std ABOVE each sample's DOI: {std_above:.4f} (log10 rho)")
    print(f"Mean learned posterior std BELOW each sample's DOI: {std_below:.4f} (log10 rho)")
    print("(Expect BELOW > ABOVE if the flow is correctly learning that deep,")
    print(" DOI-starved layers are less resolved -- with no explicit weighting")
    print(" telling it to do so.)")

# ============================================================
# Training / calibration plots (use the objects defined above)
# ============================================================
# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

axes[0].plot(history["epoch"], history["train_nll"], label="train NLL")
axes[0].plot(history["epoch"], history["val_nll"],   label="val NLL")
axes[0].axvline(best_epoch, color="k", ls=":", lw=1)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("NLL")
axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)
axes[0].set_title("NLL + posterior-mean R²")
if history["val_r2"]:
    ax0b = axes[0].twinx()
    ax0b.plot(history["val_r2_epochs"], history["val_r2"], 'g^-',
              ms=4, label="val R² (post. mean)")
    ax0b.set_ylabel("R²", color='g')
    ax0b.set_ylim(0, 1.0)
    ax0b.tick_params(axis='y', labelcolor='g')

axes[1].hist(u.ravel(), bins=20, density=True, alpha=0.75, color='steelblue')
axes[1].axhline(1.0, color='k', ls='--', lw=1, label='perfect calibration')
axes[1].set_xlabel("normalized rank of truth in posterior")
axes[1].set_ylabel("density"); axes[1].legend()
axes[1].set_title("Rank histogram (flat = calibrated)")
axes[1].grid(alpha=0.3)

z_mid = ZC
mean_std_un = (post_std * y_std[None, :]).mean(0)
axes[2].plot(z_mid, mean_std_un, 'o-', color='purple', label='learned posterior std')
if doi_te_sel is not None:
    axes[2].axvline(doi_te_sel.mean(), color='k', ls='--', lw=1.2,
                     label=f'mean DOI ({doi_te_sel.mean():.0f} m)')
    axes[2].legend(fontsize=8)
axes[2].set_xlabel("layer mid-depth (m)")
axes[2].set_ylabel("mean posterior std (log10 ρ)")
axes[2].set_title("Posterior width vs depth (vs. physics DOI)")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "cnf_training_calibration.png", dpi=130)
plt.show()

with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump({"posterior_mean_r2": float(r2_mean),
               "coverage_68_mean": float(cov68.mean()),
               "coverage_95_mean": float(cov95.mean()),
               "best_epoch": best_epoch,
               "best_val_nll": float(best_val)}, f, indent=2)
print(f"\nSaved to {OUT_DIR.resolve()}")

# ============================================================
# FIELD INFERENCE FUNCTION  (use after training)
# ============================================================
def posterior_for_field(x_lm_log_raw, x_hm_log_raw, n_samples=5000):
    """
    x_lm_log_raw : (N_LM,) log10 field LM data on the NN gate grid
    x_hm_log_raw : (N_HM,) log10 field HM data
    Returns samples of log10(rho): (n_samples, N_LAY), unstandardized.
    NOTE: no artificial noise is added here — the field data already
    contains real measurement noise.
    """
    encoder.eval(); flow.eval()
    xlm = torch.from_numpy(
        ((x_lm_log_raw - xlm_mean) / xlm_std)[None, None, :].astype(np.float32)
        ).to(DEVICE)
    xhm = torch.from_numpy(
        ((x_hm_log_raw - xhm_mean) / xhm_std)[None, None, :].astype(np.float32)
        ).to(DEVICE)
    with torch.no_grad():
        ctx = encoder(xlm, xhm)
        samp = flow(ctx).sample((n_samples,)).squeeze(1).cpu().numpy()
    return samp * y_std[None, :] + y_mean[None, :]

print("""
=================================================================
USAGE ON FIELD DATA (after this script finishes):

    samples = posterior_for_field(np.log10(d_lm_field),
                                  np.log10(d_hm_field),
                                  n_samples=5000)
    med  = np.median(samples, axis=0)
    p16, p84 = np.percentile(samples, [16, 84], axis=0)     # 68% CI
    p2 , p98 = np.percentile(samples, [2.5, 97.5], axis=0)  # 95% CI
=================================================================
""")
