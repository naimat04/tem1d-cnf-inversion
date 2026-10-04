import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
import config                                                   # all paths live in config.py
os.chdir(config.WORK_DIR)                                       # outputs (.npz/.png/checkpoints) land in outputs/

from tem1d_cnf.tem1d_wrapper import tem1d_forward
from tem1d_cnf.instrument import gate_t_lm, gate_t_hm, twave_lm, awave_lm, twave_hm, awave_hm
from tem1d_cnf.forward import forward_on_gates, interp_resp_and_jac

# ============================================================
#  Apply the trained CNF to one field sounding
#
#  Reads one sounding from "June data.usf" and produces:
#    - posterior median resistivity profile
#    - 68 % and 95 % credible-interval bands vs depth
#    - comparison with the Aarhus SPIA smooth inversion of the same
#      sounding (Line001.xyz, matched by station name; falls back to the
#      nearest station by coordinates, with a printed warning)
#    - forward consistency of the posterior median and of random posterior
#      samples (do the posterior models actually fit the data?)
#    - marginal posterior histograms for selected layers
#
#  The depth grid is rebuilt from the Aarhus geometry (constant growth factor
#  per layer, starting at 5.10622 m; N_LAY from norm_stats.npz). It must
#  match the grid used by 01_generate_training_data.py. Loading depn/thk
#  from the training .npz is a safer alternative (see the commented-out block
#  below).
#
#  Requires:  pip install zuko pandas
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import re
import zuko

# ============================================================
# CONFIG
# ============================================================
# NOTE: file locations come from config.py. If June data.usf / Line001.xyz live
# somewhere else, set the TEM_DATA_DIR environment variable (or edit config.py).
USF_PATH   = str(config.USF_PATH)
OCCAM_XYZ  = str(config.XYZ_PATH)
CNF_DIR    = config.CNF_DIR
for _p in (USF_PATH, OCCAM_XYZ):
    if not Path(_p).exists():
        raise FileNotFoundError(f"Field data file not found: {_p}. The data are not distributed; see data/README.md (or set TEM_DATA_DIR).")
TARGET_SOUNDING = 1            # 1-40, matches /SOUNDING_NUMBER in the .usf
FIELD_SHIFT     = 1.04         # matches /FIELD_SHIFT_FACTOR in June_data.usf
N_POST_SAMPLES  = 5000
N_FWD_SAMPLES   = 20          # posterior samples to push through the forward
CONTEXT_DIM     = 128
FLOW_TRANSFORMS = 5
FLOW_HIDDEN     = [256, 256]
FLOW_BINS       = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real Aarhus depth-grid geometry, measured off Line001.xyz (constant
# across all 40 stations -- std of the ratio is 3.8e-5). Used to rebuild
# the CNF's depth grid below, generically for whatever N_LAY it was
# trained with. Matches THK_TOP in 01_generate_training_data.py.
AARHUS_THK_TOP        = 5.10622
AARHUS_GROWTH_FACTOR  = 1.128994

# ============================================================
# .USF PARSER (extended to also capture the target sounding's name
# and (x, y), needed to look it up in Line001.xyz below)
# ============================================================
def parse_usf(path, target_sounding=1):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    result = {}
    current_sounding = None; current_sweep = None
    in_table = False; tbl = []
    pending_name = None
    for ln in lines:
        s = ln.strip()
        if s.startswith('/SOUNDING_NAME:'):
            pending_name = s.split(':', 1)[1].strip()
        elif s.startswith('/SOUNDING_NUMBER:'):
            current_sounding = int(s.split(':', 1)[1].strip()); current_sweep = None
            if current_sounding == target_sounding:
                result['sounding_name'] = pending_name
        elif s.startswith('/LOCATION:'):
            if current_sounding == target_sounding:
                xy = [float(v) for v in s.split(':', 1)[1].split(',')[:2]]
                result['x'], result['y'] = xy[0], xy[1]
        elif s.startswith('/SWEEP_NUMBER:'):
            current_sweep = int(s.split(':', 1)[1].strip()); in_table = False; tbl = []
        elif s.startswith('/END'):
            if in_table and current_sounding == target_sounding:
                arr = np.array(tbl, dtype=float)
                key = 'LM' if current_sweep == 1 else 'HM'
                result[key] = {'t': arr[:, 0], 'v': arr[:, 1],
                               'e': arr[:, 2], 'q': arr[:, 3].astype(int)}
            in_table = False; tbl = []
        elif s.startswith('TIME'):
            in_table = True
        elif in_table and s and not s.startswith('/') and not s.startswith('//'):
            parts = re.split(r'[\s,]+', s)
            if len(parts) >= 4:
                try:
                    tbl.append([float(parts[0]), float(parts[1]),
                                float(parts[2]), int(parts[3])])
                except ValueError:
                    pass
    return result

print(f"Parsing sounding {TARGET_SOUNDING}...")
usf = parse_usf(USF_PATH, TARGET_SOUNDING)
usf['LM']['v'] = usf['LM']['v'] / FIELD_SHIFT
usf['HM']['v'] = usf['HM']['v'] / FIELD_SHIFT
print(f"  name '{usf.get('sounding_name')}'  at ({usf.get('x')}, {usf.get('y')})")
print(f"  LM {len(usf['LM']['t'])} gates,  HM {len(usf['HM']['t'])} gates")

def interp_field(f_t, f_v, target_t):
    mask = f_v > 0
    lt = np.log10(f_t[mask]); lv = np.log10(f_v[mask])
    tq = np.clip(np.log10(target_t), lt[0], lt[-1])
    return 10.0 ** np.interp(tq, lt, lv)

d_lm_field = interp_field(usf['LM']['t'], usf['LM']['v'], gate_t_lm)
d_hm_field = interp_field(usf['HM']['t'], usf['HM']['v'], gate_t_hm)

# ============================================================
# CNF MODEL CLASSES (must match training script)
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
        return self.pool(x).squeeze(-1)

class ContextEncoder(nn.Module):
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
        return self.proj(f)

# ============================================================
# LOAD CNF
# ============================================================
stats = np.load(CNF_DIR / "norm_stats.npz")
xlm_mean, xlm_std = stats["xlm_mean"], stats["xlm_std"]
xhm_mean, xhm_std = stats["xhm_mean"], stats["xhm_std"]
y_mean,   y_std   = stats["y_mean"],   stats["y_std"]
N_LAY = len(y_mean); N_LM = len(xlm_mean); N_HM = len(xhm_mean)

encoder = ContextEncoder(CONTEXT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(CNF_DIR / "encoder_best.pt", map_location=DEVICE))
encoder.eval()

flow = zuko.flows.NSF(features=N_LAY, context=CONTEXT_DIM,
                      transforms=FLOW_TRANSFORMS, bins=FLOW_BINS,
                      hidden_features=FLOW_HIDDEN).to(DEVICE)
flow.load_state_dict(torch.load(CNF_DIR / "flow_best.pt", map_location=DEVICE))
flow.eval()
print("CNF loaded.")

# ============================================================
# SAMPLE THE POSTERIOR FOR THE FIELD SOUNDING
# ============================================================
x_lm_log = np.log10(d_lm_field).astype(np.float32)
x_hm_log = np.log10(d_hm_field).astype(np.float32)

xlm_t = torch.from_numpy(((x_lm_log - xlm_mean) / xlm_std)[None, None, :]).to(DEVICE)
xhm_t = torch.from_numpy(((x_hm_log - xhm_mean) / xhm_std)[None, None, :]).to(DEVICE)

with torch.no_grad():
    ctx = encoder(xlm_t, xhm_t)
    samples_n = flow(ctx).sample((N_POST_SAMPLES,)).squeeze(1).cpu().numpy()

samples = samples_n * y_std[None, :] + y_mean[None, :]   # log10(rho), (N_POST_SAMPLES, N_LAY)
print(f"Drew {N_POST_SAMPLES} posterior samples.")

med       = np.median(samples, axis=0)
p16, p84  = np.percentile(samples, [16, 84],   axis=0)
p2p5, p97 = np.percentile(samples, [2.5, 97.5], axis=0)
print(f"Posterior median ρ range: {10**med.min():.3f} - {10**med.max():.3f} Ω·m")

# ============================================================
# DEPTH GRID  (fixed -- see note 2 at the top of this file)
# ============================================================
thk = AARHUS_THK_TOP * AARHUS_GROWTH_FACTOR ** np.arange(N_LAY - 1)
thk = np.append(thk, thk[-1])
depn_nn = np.concatenate([[0.0], np.cumsum(thk)[:-1]])
z_top = depn_nn
z_bot = np.concatenate([depn_nn[1:], [depn_nn[-1] + thk[-1]]])
z_mid = 0.5 * (z_top + z_bot)
print(f"CNF depth grid: {N_LAY} layers, 0 -> {z_bot[-1]:.1f} m")

# --- safer alternative: load the exact grid used at training time ---
# grid = np.load(config.TRAIN_NPZ)
# depn_nn, thk = grid["depn"].astype(float), grid["thk"].astype(float)
# z_top = depn_nn
# z_bot = np.concatenate([depn_nn[1:], [depn_nn[-1] + thk[-1]]])
# z_mid = 0.5 * (z_top + z_bot)

# ============================================================
# OCCAM / XYZ  --  look up this sounding's smooth-inversion profile
# ============================================================
_XYZ_COLS = ['ID','Line_No','Layer_No','Station_Name','Model_Name','X','Y',
             'Elevation_Cell','Resistivity','Resistivity_STD','Conductivity',
             'Depth_top','Depth_bottom','Thickness','Thickness_STD']

def parse_aarhus_xyz(path):
    """Parse a SPIA .xyz export into station_name -> profile dict."""
    with open(path, encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()
    header_idx = next(i for i, l in enumerate(raw_lines) if l.strip().startswith('/ ID'))
    df = pd.read_csv(path, skiprows=header_idx + 1, sep=r'\s+',
                      names=_XYZ_COLS, engine='python')
    stations = {}
    for name, g in df.groupby('Station_Name', sort=False):
        g = g.sort_values('Layer_No')
        stations[name] = dict(
            rho=g['Resistivity'].to_numpy(float),
            top=g['Depth_top'].to_numpy(float),
            bot=g['Depth_bottom'].to_numpy(float),
            thk=g['Thickness'].to_numpy(float),
            x=g['X'].iloc[0], y=g['Y'].iloc[0],
        )
    return stations

def lookup_occam_profile(stations, sounding_name, x, y):
    if sounding_name in stations:
        return stations[sounding_name]
    print(f"  WARNING: '{sounding_name}' not found in xyz by name -- "
          f"falling back to nearest station by coordinates.")
    best, best_d = None, np.inf
    for name, prof in stations.items():
        d = (prof['x'] - x) ** 2 + (prof['y'] - y) ** 2
        if d < best_d:
            best, best_d = name, d
    print(f"  matched to '{best}' ({np.sqrt(best_d):.1f} m away)")
    return stations[best]

xyz_stations = parse_aarhus_xyz(OCCAM_XYZ)
occam_prof = lookup_occam_profile(xyz_stations, usf.get('sounding_name'),
                                  usf.get('x'), usf.get('y'))

occam_rho = occam_prof['rho']
occam_top = occam_prof['top']
occam_bot = occam_prof['bot']
bad = ~np.isfinite(occam_bot) | (occam_bot <= 0)
if bad.any():
    thk_valid = occam_prof['thk'][np.isfinite(occam_prof['thk']) & (occam_prof['thk'] > 0)]
    fb = np.median(thk_valid) if thk_valid.size else 5.0
    thk_use = np.where(np.isfinite(occam_prof['thk']) & (occam_prof['thk'] > 0),
                       occam_prof['thk'], fb)
    occam_bot = np.where(bad, np.cumsum(thk_use), occam_bot)
occam_rho_clip = np.clip(occam_rho, 1e-3, 1e5)
occam_mid = 0.5 * (occam_top + occam_bot)

# --- does Occam fall inside the CNF credible band? -----------
log_occam_on_grid = np.interp(z_mid, occam_mid, np.log10(occam_rho_clip))
inside68 = (log_occam_on_grid > p16) & (log_occam_on_grid < p84)
inside95 = (log_occam_on_grid > p2p5) & (log_occam_on_grid < p97)
n_top = max(1, int(round(N_LAY * 0.6)))     # "well-resolved" ~ shallow 60% of layers
print(f"\nOccam-inside-band check (all {N_LAY} layers):")
print(f"  inside 68% CI: {inside68.sum()}/{N_LAY} layers")
print(f"  inside 95% CI: {inside95.sum()}/{N_LAY} layers")
print(f"  Top {n_top} layers (well-resolved): 68% -> {inside68[:n_top].sum()}/{n_top},"
      f"  95% -> {inside95[:n_top].sum()}/{n_top}")

# ============================================================
# FORWARD-CONSISTENCY: posterior median AND random samples
# ============================================================
def forward_pred(rho):
    p_lm, _, _ = forward_on_gates(rho, depn_nn, twave_lm, awave_lm,
                                  repfreq=512.82, gate_t=gate_t_lm, want_jac=False)
    p_hm, _, _ = forward_on_gates(rho, depn_nn, twave_hm, awave_hm,
                                  repfreq=50.0, gate_t=gate_t_hm, want_jac=False)
    return p_lm, p_hm

def logrmse(p, o): return float(np.sqrt(np.mean((np.log10(p) - np.log10(o))**2)))
def combi(a, b):   return np.sqrt((a**2 + b**2) / 2)

print("\nForwarding posterior median...")
d_lm_med, d_hm_med = forward_pred(10.0 ** med)
r_med = (logrmse(d_lm_med, d_lm_field), logrmse(d_hm_med, d_hm_field))

print(f"Forwarding {N_FWD_SAMPLES} random posterior samples...")
rng = np.random.default_rng(0)
sub = rng.choice(N_POST_SAMPLES, size=N_FWD_SAMPLES, replace=False)
fwd_lm = np.zeros((N_FWD_SAMPLES, len(gate_t_lm)))
fwd_hm = np.zeros((N_FWD_SAMPLES, len(gate_t_hm)))
r_samp = []
for k, i in enumerate(sub):
    try:
        plm, phm = forward_pred(10.0 ** samples[i])
        fwd_lm[k] = plm; fwd_hm[k] = phm
        r_samp.append(combi(logrmse(plm, d_lm_field), logrmse(phm, d_hm_field)))
    except Exception:
        fwd_lm[k] = np.nan; fwd_hm[k] = np.nan
r_samp = np.array(r_samp)

# Occam forward for reference -- resampled onto the CNF's own depth
# grid (log_occam_on_grid), NOT a naive occam_rho[:N_LAY] truncation:
# the real xyz profile always has 20 layers regardless of what N_LAY
# the CNF uses, so truncating instead of resampling silently forwards
# the wrong physical depths whenever they differ (see note 3 at top).
occam_rho_use = np.clip(10.0 ** log_occam_on_grid, 1e-2, 1e4)
d_lm_O, d_hm_O = forward_pred(occam_rho_use)
r_O = (logrmse(d_lm_O, d_lm_field), logrmse(d_hm_O, d_hm_field))

print("\n" + "=" * 66)
print("FORWARD-CONSISTENCY (log10 RMSE)")
print("=" * 66)
print(f"{'':>26} {'LM':>9} {'HM':>9} {'combined':>10}")
print(f"{'CNF posterior median':>26} {r_med[0]:9.4f} {r_med[1]:9.4f} {combi(*r_med):10.4f}")
print(f"{'CNF samples (mean±std)':>26} {'':>9} {'':>9} "
      f"{np.nanmean(r_samp):7.4f}±{np.nanstd(r_samp):.4f}")
print(f"{'Occam':>26} {r_O[0]:9.4f} {r_O[1]:9.4f} {combi(*r_O):10.4f}")

# ============================================================
# PLOTS
# ============================================================
fig = plt.figure(figsize=(17, 10))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

# --- (0,0) posterior profile with credible bands ---
ax1 = fig.add_subplot(gs[0, 0])
# bands drawn as filled steps on the mid-depth grid
ax1.fill_betweenx(z_mid, 10**p2p5, 10**p97, color='skyblue', alpha=0.35,
                  label='95% CI')
ax1.fill_betweenx(z_mid, 10**p16, 10**p84, color='steelblue', alpha=0.45,
                  label='68% CI')
ax1.plot(10**med, z_mid, 'b-', lw=2.2, label='CNF median')
# Occam overlay (full native 20-layer profile, own depths -- not resampled)
xO = np.repeat(occam_rho_clip, 2)
yO = np.concatenate([[occam_top[0]], np.repeat(occam_bot[:-1], 2), [occam_bot[-1]]])
ax1.plot(xO, yO, 'k--', lw=1.6, label='Occam', alpha=0.8)
ax1.set_xscale('log'); ax1.invert_yaxis()
ax1.set_xlabel('Resistivity (Ω·m)'); ax1.set_ylabel('Depth (m)')
ax1.set_title(f'Bayesian posterior — {usf.get("sounding_name")} (#{TARGET_SOUNDING})')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=9)
ax1.set_xlim(0.05, 5000)      # matches the training prior's 0.1-3162 ohm.m range

# --- (0,1) LM fit with posterior sample spaghetti ---
ax2 = fig.add_subplot(gs[0, 1])
for k in range(N_FWD_SAMPLES):
    if np.all(np.isfinite(fwd_lm[k])):
        ax2.loglog(gate_t_lm, fwd_lm[k], '-', color='steelblue',
                   alpha=0.25, lw=0.8)
ax2.loglog(usf['LM']['t'], usf['LM']['v'], 'k-', lw=2.2, label='Field')
ax2.loglog(gate_t_lm, d_lm_med, 'b--', lw=1.8, label='posterior median → fwd')
ax2.loglog(gate_t_lm, d_lm_O, 'g:', lw=1.5, label='Occam → fwd', alpha=0.8)
ax2.set_xlabel('t (s)'); ax2.set_ylabel('|dB/dt|')
ax2.set_title(f'LM — spaghetti = {N_FWD_SAMPLES} posterior samples')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=8)

# --- (0,2) HM fit with spaghetti ---
ax3 = fig.add_subplot(gs[0, 2])
for k in range(N_FWD_SAMPLES):
    if np.all(np.isfinite(fwd_hm[k])):
        ax3.loglog(gate_t_hm, fwd_hm[k], '-', color='steelblue',
                   alpha=0.25, lw=0.8)
ax3.loglog(usf['HM']['t'], usf['HM']['v'], 'k-', lw=2.2, label='Field')
ax3.loglog(gate_t_hm, d_hm_med, 'b--', lw=1.8, label='posterior median → fwd')
ax3.loglog(gate_t_hm, d_hm_O, 'g:', lw=1.5, label='Occam → fwd', alpha=0.8)
ax3.set_xlabel('t (s)'); ax3.set_ylabel('|dB/dt|')
ax3.set_title('HM — spaghetti = posterior samples')
ax3.grid(alpha=0.3, which='both'); ax3.legend(fontsize=8)

# --- (1,0-2) marginal posteriors for 3 representative layers ---
# picked as roughly shallow/mid/deep fractions of N_LAY rather than
# hardcoded indices, since N_LAY now depends on how the CNF was trained
pick_layers = sorted(set(
    int(np.clip(round(N_LAY * f), 0, N_LAY - 1)) for f in (0.15, 0.45, 0.85)
))
for j, L in enumerate(pick_layers):
    ax = fig.add_subplot(gs[1, j])
    ax.hist(samples[:, L], bins=60, density=True, alpha=0.7,
            color='steelblue')
    ax.axvline(med[L], color='b', lw=2, label='median')
    ax.axvline(log_occam_on_grid[L], color='k', ls='--', lw=1.8, label='Occam')
    ax.axvline(p16[L], color='b', ls=':', lw=1)
    ax.axvline(p84[L], color='b', ls=':', lw=1)
    ax.set_xlabel('log10(ρ)')
    if j == 0: ax.set_ylabel('posterior density')
    ax.set_title(f'Layer {L+1}  (z≈{z_mid[L]:.0f} m)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle(f'CNF Bayesian inversion — {usf.get("sounding_name")} (#{TARGET_SOUNDING})',
             fontsize=13)
fig.tight_layout()
fig.savefig('cnf_field_posterior.png', dpi=130)
plt.show()

# ============================================================
# HONEST SUMMARY
# ============================================================
print("\n" + "=" * 66)
print("HONEST SUMMARY")
print("=" * 66)
print(f"""
1. Occam inside 68% band: {inside68[:n_top].sum()}/{n_top} well-resolved layers
   ({'GOOD - posterior covers the reference solution' if inside68[:n_top].sum() >= 0.6*n_top
     else 'CONCERN - posterior may be overconfident or biased'})

2. Posterior-median data fit: {combi(*r_med):.4f}
   Occam data fit:            {combi(*r_O):.4f}
   ({'Median competitive with Occam' if combi(*r_med) < combi(*r_O)*1.5
     else 'Median clearly worse than Occam - posterior may be prior-dominated'})

3. Posterior-sample fits: {np.nanmean(r_samp):.4f} ± {np.nanstd(r_samp):.4f}
   If sample fits are much worse than the median fit, the posterior
   is wider than the data demands (underconfident but safe).
   If sample fits are all tight AND coverage was good in training,
   the posterior is genuinely informative.

4. Posterior width increasing with depth = physics respected.
   Check panel 1: the 68% band should visibly widen with depth.
""")
