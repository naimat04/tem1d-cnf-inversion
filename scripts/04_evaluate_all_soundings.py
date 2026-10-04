import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
import config                                                   # all paths live in config.py
os.chdir(config.WORK_DIR)                                       # outputs (.npz/.png/checkpoints) land in outputs/

from tem1d_cnf.tem1d_wrapper import tem1d_forward
from tem1d_cnf.instrument import gate_t_lm, gate_t_hm, twave_lm, awave_lm, twave_hm, awave_hm
from tem1d_cnf.forward import forward_on_gates, interp_resp_and_jac

# ============================================================
#  Apply the trained CNF to ALL soundings in "June data.usf"
#
#  Each sounding is compared with its Aarhus SPIA smooth inversion profile
#  (Line001.xyz, matched by station name; nearest station by coordinates as
#  a fallback). Outputs:
#    - per-sounding calibration counts (inversion model inside CNF band)
#    - per-sounding data fit (forward consistency, log10 RMSE)
#    - aggregate calibration and data-fit statistics
#    - 2D pseudo-section of posterior median and uncertainty width
#    - profile grid (CNF bands vs smooth inversion) and data-fit grid
#      (observed vs forwarded posterior) for every sounding
#    - selection of the most reliable soundings (last section)
#
#  LM late-time gate truncation (must mirror training): the training script
#  may drop the last LM_GATES_TO_DROP LM gates before normalisation. The
#  count is read back from norm_stats.npz rather than hard-coded, the field
#  LM gate grid is asserted to have n_lm_full gates, and only the array fed
#  to the encoder is truncated. The forward-consistency and data-fit checks
#  always use the full gate grid, since they compare against the actual
#  field curve.
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
USF_PATH   = str(config.USF_PATH)
OCCAM_XYZ  = str(config.XYZ_PATH)
USF_PATH   = str(config.USF_PATH)
OCCAM_XYZ  = str(config.XYZ_PATH)
CNF_DIR    = config.CNF_DIR
for _p in (USF_PATH, OCCAM_XYZ):
    if not Path(_p).exists():
        raise FileNotFoundError(f"Field data file not found: {_p}. The data are not distributed; see data/README.md (or set TEM_DATA_DIR).")

# All soundings in June_data.usf. Adjust the upper bound if it isn't 40.
SOUNDING_LIST = list(range(1, 41))

FIELD_SHIFT     = 1.04       # matches /FIELD_SHIFT_FACTOR in June_data.usf
N_POST          = 3000       # posterior samples per sounding
N_FWD_SAMPLES   = 20         # posterior samples pushed through the forward, per sounding
CONTEXT_DIM     = 128
FLOW_TRANSFORMS = 5
FLOW_HIDDEN     = [256, 256]
FLOW_BINS       = 8

REPFREQ_LM = 512.82
REPFREQ_HM = 50.0

# Real Aarhus depth-grid geometry, measured off Line001.xyz (constant
# across stations -- std of the ratio ~3.8e-5). Must match whatever
# scripts/01_generate_training_data.py used to build the training set.
AARHUS_THK_TOP       = 5.10622
AARHUS_GROWTH_FACTOR = 1.128994

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# .USF PARSER — all soundings, with name + location captured
# ============================================================
def parse_usf_all(path, sounding_ids):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    out = {s: {} for s in sounding_ids}
    cs = cw = None
    pending_name = None
    in_tbl = False; tbl = []
    for ln in lines:
        s = ln.strip()
        if s.startswith('/SOUNDING_NAME:'):
            pending_name = s.split(':', 1)[1].strip()
        elif s.startswith('/SOUNDING_NUMBER:'):
            cs = int(s.split(':', 1)[1].strip()); cw = None
            if cs in out:
                out[cs]['name'] = pending_name
        elif s.startswith('/LOCATION:'):
            if cs in out:
                xy = [float(v) for v in s.split(':', 1)[1].split(',')[:2]]
                out[cs]['x'], out[cs]['y'] = xy[0], xy[1]
        elif s.startswith('/SWEEP_NUMBER:'):
            cw = int(s.split(':', 1)[1].strip()); in_tbl = False; tbl = []
        elif s.startswith('/END'):
            if in_tbl and cs in out:
                arr = np.array(tbl, dtype=float)
                key = 'LM' if cw == 1 else 'HM'
                out[cs][key] = {'t': arr[:, 0], 'v': arr[:, 1] / FIELD_SHIFT,
                                'e': arr[:, 2], 'q': arr[:, 3].astype(int)}
            in_tbl = False; tbl = []
        elif s.startswith('TIME'):
            in_tbl = True
        elif in_tbl and s and not s.startswith('/') and not s.startswith('//'):
            parts = re.split(r'[\s,]+', s)
            if len(parts) >= 4:
                try:
                    tbl.append([float(parts[0]), float(parts[1]),
                               float(parts[2]), int(parts[3])])
                except ValueError:
                    pass
    return out

print(f"Parsing {len(SOUNDING_LIST)} soundings from {USF_PATH} ...")
soundings = parse_usf_all(USF_PATH, SOUNDING_LIST)
valid = [s for s in SOUNDING_LIST if 'LM' in soundings[s] and 'HM' in soundings[s]]
print(f"  parsed cleanly: {len(valid)}/{len(SOUNDING_LIST)}")
missing = set(SOUNDING_LIST) - set(valid)
if missing:
    print(f"  MISSING from .usf: {sorted(missing)}")

def interp_field(f_t, f_v, target_t):
    mask = f_v > 0
    lt = np.log10(f_t[mask]); lv = np.log10(f_v[mask])
    tq = np.clip(np.log10(target_t), lt[0], lt[-1])
    return 10.0 ** np.interp(tq, lt, lv)

# ============================================================
# OCCAM / XYZ — parse once, look up per sounding by name
# ============================================================
_XYZ_COLS = ['ID', 'Line_No', 'Layer_No', 'Station_Name', 'Model_Name', 'X', 'Y',
             'Elevation_Cell', 'Resistivity', 'Resistivity_STD', 'Conductivity',
             'Depth_top', 'Depth_bottom', 'Thickness', 'Thickness_STD']

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
        return stations[sounding_name], True
    if x is None or y is None:
        return None, False
    best, best_d = None, np.inf
    for name, prof in stations.items():
        d = (prof['x'] - x) ** 2 + (prof['y'] - y) ** 2
        if d < best_d:
            best, best_d = name, d
    if best is None:
        return None, False
    print(f"    WARNING: name '{sounding_name}' not in xyz -- fell back to "
          f"nearest station '{best}' ({np.sqrt(best_d):.1f} m away)")
    return stations[best], False

print(f"Parsing Occam/SPIA export {OCCAM_XYZ} ...")
xyz_stations = parse_aarhus_xyz(OCCAM_XYZ)
print(f"  {len(xyz_stations)} stations found")

def load_occam_for(sounding_num):
    """Returns (rho, top, bot) on the Occam's own native depth grid,
    or None if no match could be found at all."""
    name = soundings[sounding_num].get('name')
    x = soundings[sounding_num].get('x')
    y = soundings[sounding_num].get('y')
    prof, exact = lookup_occam_profile(xyz_stations, name, x, y)
    if prof is None:
        return None
    rho = prof['rho']; top = prof['top']; bot = prof['bot']
    bad = ~np.isfinite(bot) | (bot <= 0)
    if bad.any():
        thk_valid = prof['thk'][np.isfinite(prof['thk']) & (prof['thk'] > 0)]
        fb = np.median(thk_valid) if thk_valid.size else 5.0
        thk_use = np.where(np.isfinite(prof['thk']) & (prof['thk'] > 0), prof['thk'], fb)
        bot = np.where(bad, np.cumsum(thk_use), bot)
    rho_clip = np.clip(rho, 1e-3, 1e5)
    return rho_clip, top, bot

# ============================================================
# CNF ARCHITECTURE (matching training)
# ============================================================
class ResBlock1D(nn.Module):
    def __init__(self, ch, k=3, do=0.0):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(ch, ch, k, padding=p); self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=p); self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(do) if do > 0 else nn.Identity()
    def forward(self, x):
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.drop(self.act(x + r))

class Branch(nn.Module):
    def __init__(self, do=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2),
                                  nn.BatchNorm1d(32), nn.LeakyReLU(0.1, inplace=True))
        self.res1 = ResBlock1D(32, 3, do); self.res2 = ResBlock1D(32, 3, do)
        self.expand = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1),
                                    nn.BatchNorm1d(64), nn.LeakyReLU(0.1, inplace=True))
        self.res3 = ResBlock1D(64, 3, do); self.res4 = ResBlock1D(64, 3, do)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x); x = self.res1(x); x = self.res2(x)
        x = self.expand(x); x = self.res3(x); x = self.res4(x)
        return self.pool(x).squeeze(-1)

class ContextEncoder(nn.Module):
    def __init__(self, dim=128, do=0.1):
        super().__init__()
        self.lm_branch = Branch(do); self.hm_branch = Branch(do)
        self.proj = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(do),
            nn.Linear(256, dim))
    def forward(self, xlm, xhm):
        return self.proj(torch.cat([self.lm_branch(xlm), self.hm_branch(xhm)], 1))

# ============================================================
# LOAD MODEL
# ============================================================
stats = np.load(CNF_DIR / "norm_stats.npz")
xlm_mean, xlm_std = stats["xlm_mean"], stats["xlm_std"]
xhm_mean, xhm_std = stats["xhm_mean"], stats["xhm_std"]
y_mean,   y_std   = stats["y_mean"],   stats["y_std"]
N_LAY = len(y_mean)
N_LM_ENC = len(xlm_mean)   # number of LM gates the encoder actually expects

# ---- LM late-time gate truncation: read back from training, don't guess ----
if "lm_gates_to_drop" in stats.files:
    LM_GATES_TO_DROP = int(stats["lm_gates_to_drop"])
    N_LM_FULL_TRAIN  = int(stats["n_lm_full"])
else:
    # older checkpoint saved before truncation was introduced
    LM_GATES_TO_DROP = 0
    N_LM_FULL_TRAIN  = N_LM_ENC
    print("  WARNING: norm_stats.npz has no lm_gates_to_drop field -- assuming "
          "this checkpoint was trained on the full LM gate set (0 dropped). "
          "If that's wrong, field predictions from this script will be silently off.")

print(f"LM truncation from training: dropping last {LM_GATES_TO_DROP} of "
      f"{N_LM_FULL_TRAIN} LM gates  ->  encoder expects {N_LM_ENC} LM gates")

encoder = ContextEncoder(CONTEXT_DIM).to(DEVICE)
encoder.load_state_dict(torch.load(CNF_DIR / "encoder_best.pt", map_location=DEVICE))
encoder.eval()

flow = zuko.flows.NSF(features=N_LAY, context=CONTEXT_DIM,
                      transforms=FLOW_TRANSFORMS, bins=FLOW_BINS,
                      hidden_features=FLOW_HIDDEN).to(DEVICE)
flow.load_state_dict(torch.load(CNF_DIR / "flow_best.pt", map_location=DEVICE))
flow.eval()
print(f"CNF loaded. N_LAY={N_LAY}")

# Sanity check: the field/forward pipeline's own gate_t_lm (full grid, used
# for the physical forward model and the data-fit checks below) must have
# exactly N_LM_FULL_TRAIN gates, or the "drop last N" indices below would
# be cutting the wrong gates entirely.
assert len(gate_t_lm) == N_LM_FULL_TRAIN, (
    f"gate_t_lm from the pipeline has {len(gate_t_lm)} gates but the model "
    f"was trained expecting a full LM grid of {N_LM_FULL_TRAIN} gates "
    f"(before dropping the last {LM_GATES_TO_DROP}). The field data's gate "
    f"grid and the training data's gate grid have drifted out of sync -- "
    f"fix that before trusting anything below."
)

# ============================================================
# DEPTH GRID — real Aarhus geometry (NOT the old geomspace placeholder)
# ============================================================
thk = AARHUS_THK_TOP * AARHUS_GROWTH_FACTOR ** np.arange(N_LAY - 1)
thk = np.append(thk, thk[-1])
depn_nn = np.concatenate([[0.0], np.cumsum(thk)[:-1]])
z_top = depn_nn
z_bot = np.concatenate([depn_nn[1:], [depn_nn[-1] + thk[-1]]])
z_mid = 0.5 * (z_top + z_bot)
print(f"CNF depth grid: {N_LAY} layers, 0 -> {z_bot[-1]:.1f} m")

# --- alternative: load the exact grid used at training time ---
# grid = np.load(config.TRAIN_NPZ)
# depn_nn, thk = grid["depn"].astype(float), grid["thk"].astype(float)
# z_top = depn_nn
# z_bot = np.concatenate([depn_nn[1:], [depn_nn[-1] + thk[-1]]])
# z_mid = 0.5 * (z_top + z_bot)

# ============================================================
# FORWARD HELPERS
#   (always run on the FULL gate_t_lm grid. Dropping late-time
#    LM gates only affects what the *encoder* sees, never the physical
#    forward model or the data-fit comparisons below.)
# ============================================================
def forward_pred(rho_lin):
    p_lm, _, _ = forward_on_gates(rho_lin, depn_nn, twave_lm, awave_lm,
                                  repfreq=REPFREQ_LM, gate_t=gate_t_lm, want_jac=False)
    p_hm, _, _ = forward_on_gates(rho_lin, depn_nn, twave_hm, awave_hm,
                                  repfreq=REPFREQ_HM, gate_t=gate_t_hm, want_jac=False)
    return p_lm, p_hm

def logrmse(p, o):
    p = np.asarray(p); o = np.asarray(o)
    m = (p > 0) & (o > 0)
    return float(np.sqrt(np.mean((np.log10(p[m]) - np.log10(o[m]))**2)))

def combi(a, b):
    return float(np.sqrt((a**2 + b**2) / 2))

# ============================================================
# PER-SOUNDING INFERENCE (+ DATA FIT)
# ============================================================
results = {}
rng = np.random.default_rng(0)
n_top = max(1, int(round(N_LAY * 0.6)))   # "well-resolved" ~ shallow 60% of layers
print(f"\nRunning CNF on {len(valid)} soundings...")

for s in valid:
    # lm_data / hm_data stay on the FULL gate grid -- used for the physical
    # data-fit checks later. Only the encoder gets the truncated LM slice.
    lm_data = interp_field(soundings[s]['LM']['t'], soundings[s]['LM']['v'], gate_t_lm)
    hm_data = interp_field(soundings[s]['HM']['t'], soundings[s]['HM']['v'], gate_t_hm)

    lm_data_enc = lm_data[:-LM_GATES_TO_DROP] if LM_GATES_TO_DROP > 0 else lm_data
    assert len(lm_data_enc) == N_LM_ENC, (
        f"sounding {s}: truncated LM length {len(lm_data_enc)} != encoder's "
        f"expected {N_LM_ENC}"
    )

    x_lm_log = np.log10(lm_data_enc).astype(np.float32)
    x_hm_log = np.log10(hm_data).astype(np.float32)
    xlm = torch.from_numpy(((x_lm_log - xlm_mean) / xlm_std)[None, None, :]).to(DEVICE)
    xhm = torch.from_numpy(((x_hm_log - xhm_mean) / xhm_std)[None, None, :]).to(DEVICE)

    with torch.no_grad():
        ctx = encoder(xlm, xhm)
        samp_n = flow(ctx).sample((N_POST,)).squeeze(1).cpu().numpy()
    samples = samp_n * y_std[None, :] + y_mean[None, :]     # (N_POST, N_LAY) log10 rho

    med = np.median(samples, axis=0)
    p16, p84  = np.percentile(samples, [16, 84], axis=0)
    p2p5, p97 = np.percentile(samples, [2.5, 97.5], axis=0)

    # ---- DATA FIT: forward the posterior median -------------------------
    d_lm_med, d_hm_med = forward_pred(10.0 ** med)
    fit_lm   = logrmse(d_lm_med, lm_data)
    fit_hm   = logrmse(d_hm_med, hm_data)
    fit_comb = combi(fit_lm, fit_hm)

    # ---- DATA FIT: forward a subset of posterior samples -----------------
    sub = rng.choice(N_POST, size=min(N_FWD_SAMPLES, N_POST), replace=False)
    fwd_lm = np.full((len(sub), len(gate_t_lm)), np.nan)
    fwd_hm = np.full((len(sub), len(gate_t_hm)), np.nan)
    r_samp = []
    for k, i in enumerate(sub):
        try:
            plm, phm = forward_pred(10.0 ** samples[i])
            fwd_lm[k] = plm; fwd_hm[k] = phm
            r_samp.append(combi(logrmse(plm, lm_data), logrmse(phm, hm_data)))
        except Exception:
            pass
    r_samp = np.array(r_samp)

    rec = dict(name=soundings[s].get('name'), x=soundings[s].get('x'), y=soundings[s].get('y'),
               med=med, p16=p16, p84=p84, p2p5=p2p5, p97=p97,
               samples=samples, lm=lm_data, hm=hm_data,
               d_lm_med=d_lm_med, d_hm_med=d_hm_med,
               fwd_lm=fwd_lm, fwd_hm=fwd_hm,
               fit_lm=fit_lm, fit_hm=fit_hm, fit_comb=fit_comb,
               fit_samp_mean=float(np.nanmean(r_samp)) if r_samp.size else np.nan,
               fit_samp_std=float(np.nanstd(r_samp)) if r_samp.size else np.nan,
               raw_lm_t=soundings[s]['LM']['t'], raw_lm_v=soundings[s]['LM']['v'],
               raw_hm_t=soundings[s]['HM']['t'], raw_hm_v=soundings[s]['HM']['v'])

    occam = load_occam_for(s)
    if occam is None:
        results[s] = rec
        results[s].update(dict(inside68=None, inside95=None))
        print(f"  sounding {s:2d} ({rec['name']}): no Occam match | "
              f"fit med {fit_comb:.4f}  samp {rec['fit_samp_mean']:.4f}")
        continue

    occam_rho, occam_top, occam_bot = occam
    occam_mid = 0.5 * (occam_top + occam_bot)
    log_occam_grid = np.interp(z_mid, occam_mid, np.log10(occam_rho))
    inside68 = (log_occam_grid > p16) & (log_occam_grid < p84)
    inside95 = (log_occam_grid > p2p5) & (log_occam_grid < p97)

    # ---- DATA FIT: forward Occam, resampled onto the CNF's own grid -----
    # (never a naive occam_rho[:N_LAY] truncation -- see header note)
    d_lm_O, d_hm_O = forward_pred(10.0 ** log_occam_grid)
    fitO_comb = combi(logrmse(d_lm_O, lm_data), logrmse(d_hm_O, hm_data))

    rec.update(dict(occam_rho=occam_rho, occam_top=occam_top, occam_bot=occam_bot,
                    log_occam_grid=log_occam_grid,
                    inside68=inside68, inside95=inside95,
                    d_lm_O=d_lm_O, d_hm_O=d_hm_O, fit_occam=fitO_comb))
    results[s] = rec
    print(f"  sounding {s:2d} ({rec['name']}): 68% cov {inside68[:n_top].sum():2d}/{n_top}, "
          f"95% cov {inside95[:n_top].sum():2d}/{n_top} | "
          f"fit med {fit_comb:.4f}  samp {rec['fit_samp_mean']:.4f}  "
          f"Occam {fitO_comb:.4f}")

# ============================================================
# AGGREGATE CALIBRATION
# ============================================================
print("\n" + "=" * 72)
print(f"AGGREGATE CALIBRATION (over all soundings, top {n_top} well-resolved layers)")
print("=" * 72)

with_occam = [s for s in valid if results[s].get('inside68') is not None]
if with_occam:
    all_in68 = np.array([results[s]['inside68'] for s in with_occam])   # (S, N_LAY)
    all_in95 = np.array([results[s]['inside95'] for s in with_occam])

    cov68_layer = all_in68.mean(0)
    cov95_layer = all_in95.mean(0)

    print(f"{'layer':>5} {'depth':>7} {'68% cov':>10} {'95% cov':>10}")
    for i in range(N_LAY):
        flag = ""
        if 0.55 < cov68_layer[i] < 0.80 and 0.85 < cov95_layer[i] < 1.0:
            flag = "  ok"
        elif cov68_layer[i] < 0.55: flag = "  overconfident"
        elif cov68_layer[i] > 0.80: flag = "  underconfident"
        print(f"{i+1:5d} {z_mid[i]:7.1f} {cov68_layer[i]:10.3f} "
              f"{cov95_layer[i]:10.3f}{flag}")

    print(f"\nMean coverage over top {n_top} (well-resolved) layers:")
    print(f"  68% CI: {cov68_layer[:n_top].mean():.3f}  (target 0.68)")
    print(f"  95% CI: {cov95_layer[:n_top].mean():.3f}  (target 0.95)")
    print(f"\nMean coverage over ALL {N_LAY} layers:")
    print(f"  68% CI: {cov68_layer.mean():.3f}")
    print(f"  95% CI: {cov95_layer.mean():.3f}")
else:
    print("No sounding had a matched Occam profile -- skipping calibration table.")

# ============================================================
# AGGREGATE DATA FIT
# ============================================================
print("\n" + "=" * 72)
print("AGGREGATE DATA FIT (forward-consistency, log10 RMSE)")
print("=" * 72)
print(f"{'sounding':>9} {'name':>14} {'median':>9} {'samp mean':>11} {'samp std':>9} {'Occam':>9}")
med_fits, samp_fits, occ_fits = [], [], []
for s in valid:
    R = results[s]
    o = R.get('fit_occam', np.nan)
    med_fits.append(R['fit_comb']); samp_fits.append(R['fit_samp_mean'])
    if np.isfinite(o): occ_fits.append(o)
    nm = str(R.get('name'))[:14]
    print(f"{s:9d} {nm:>14} {R['fit_comb']:9.4f} {R['fit_samp_mean']:11.4f} "
          f"{R['fit_samp_std']:9.4f} "
          f"{('%.4f' % o) if np.isfinite(o) else '   --':>9}")

print("-" * 66)
print(f"{'MEAN':>9} {'':>14} {np.nanmean(med_fits):9.4f} {np.nanmean(samp_fits):11.4f} "
      f"{'':>9} {(np.nanmean(occ_fits) if occ_fits else np.nan):9.4f}")
print("""
Read this ALONGSIDE the coverage table above:
  * Low median fit + LOW coverage  -> overconfident (fits data, misses truth
    band) -- the classic amortized-SBI under-dispersion signature.
  * Sample fits >> median fit       -> posterior wider than data demands.
  * Median fit competitive w/ Occam -> posterior is data-informed, not
    prior-dominated.
""")

# ============================================================
# 2D SECTION PLOTS (median resistivity + posterior width)
# x-axis uses each sounding's real x-coordinate when available,
# falling back to sounding number otherwise.
# ============================================================
S = np.array(valid); n_S = len(S)
med_2d   = np.stack([results[s]['med']       for s in valid], axis=1)   # (N_LAY, S)
width_2d = np.stack([results[s]['p84'] - results[s]['p16'] for s in valid], axis=1)
occam_2d = np.full((N_LAY, n_S), np.nan)
for k, s in enumerate(valid):
    if 'log_occam_grid' in results[s]:
        occam_2d[:, k] = results[s]['log_occam_grid']

has_x = all(results[s].get('x') is not None for s in valid)
x_pos = np.array([results[s]['x'] for s in valid], dtype=float) if has_x \
        else np.array(valid, dtype=float)
order = np.argsort(x_pos)   # section should run along the line, not sounding order
x_pos_sorted = x_pos[order]
med_2d_sorted   = med_2d[:, order]
width_2d_sorted = width_2d[:, order]
occam_2d_sorted = occam_2d[:, order]

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
vmin, vmax = np.nanmin(med_2d_sorted), np.nanmax(med_2d_sorted)

im0 = axes[0].pcolormesh(x_pos_sorted, z_mid, med_2d_sorted, cmap='viridis_r',
                         shading='auto', vmin=vmin, vmax=vmax)
axes[0].invert_yaxis(); axes[0].set_ylabel("depth (m)")
axes[0].set_title("CNF posterior MEDIAN (log10 ρ)")
plt.colorbar(im0, ax=axes[0], label="log10 ρ")

im1 = axes[1].pcolormesh(x_pos_sorted, z_mid, occam_2d_sorted, cmap='viridis_r',
                         shading='auto', vmin=vmin, vmax=vmax)
axes[1].invert_yaxis(); axes[1].set_ylabel("depth (m)")
axes[1].set_title("Occam / SPIA smooth inversion (log10 ρ)")
plt.colorbar(im1, ax=axes[1], label="log10 ρ")

im2 = axes[2].pcolormesh(x_pos_sorted, z_mid, width_2d_sorted, cmap='Reds', shading='auto')
axes[2].invert_yaxis()
axes[2].set_xlabel("x (m)" if has_x else "sounding number"); axes[2].set_ylabel("depth (m)")
axes[2].set_title("Posterior 68% CI WIDTH (log10 ρ)  -- narrow=well resolved")
plt.colorbar(im2, ax=axes[2], label="width (log10 units)")

fig.tight_layout(); fig.savefig("cnf_sections.png", dpi=130); plt.show()

# ============================================================
# INDIVIDUAL PROFILE GRID (every sounding)
# ============================================================
nS = len(valid)
nc = min(nS, 5); nr = int(np.ceil(nS / nc))
fig2, axf = plt.subplots(nr, nc, figsize=(3*nc, 4*nr), sharey=True)
axf = np.array(axf).reshape(nr, nc)

for k, s in enumerate(valid):
    ax = axf[k // nc, k % nc]
    R = results[s]
    ax.fill_betweenx(z_mid, 10**R['p2p5'], 10**R['p97'],
                     color='skyblue', alpha=0.35, label='95% CI')
    ax.fill_betweenx(z_mid, 10**R['p16'],  10**R['p84'],
                     color='steelblue', alpha=0.45, label='68% CI')
    ax.plot(10**R['med'], z_mid, 'b-', lw=1.8, label='CNF median')
    if 'occam_rho' in R:
        xO = np.repeat(R['occam_rho'], 2)
        yO = np.concatenate([[R['occam_top'][0]],
                             np.repeat(R['occam_bot'][:-1], 2),
                             [R['occam_bot'][-1]]])
        ax.plot(xO, yO, 'k--', lw=1.2, alpha=0.8, label='Occam')
    ax.set_xscale('log'); ax.invert_yaxis()
    ax.set_title(f"S{s} {R.get('name','')}", fontsize=8)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0.05, 5000)
    if k == 0: ax.legend(fontsize=7)

for k in range(nS, nr*nc):
    axf[k // nc, k % nc].set_visible(False)
axf[-1, 0].set_xlabel("ρ (Ω·m)"); axf[0, 0].set_ylabel("depth (m)")
fig2.suptitle("CNF posterior — every sounding")
fig2.tight_layout(); fig2.savefig("cnf_soundings_grid.png", dpi=130); plt.show()

# ============================================================
# DATA-SPACE FIT GRID (observed vs forwarded posterior, every sounding)
# ============================================================
fig3, axd = plt.subplots(nr, nc, figsize=(3.2*nc, 4*nr), sharex=True, sharey=True)
axd = np.array(axd).reshape(nr, nc)

for k, s in enumerate(valid):
    ax = axd[k // nc, k % nc]
    R = results[s]

    for j in range(R['fwd_lm'].shape[0]):
        if np.all(np.isfinite(R['fwd_lm'][j])):
            ax.loglog(gate_t_lm, R['fwd_lm'][j], '-', color='steelblue',
                      alpha=0.20, lw=0.7)
        if np.all(np.isfinite(R['fwd_hm'][j])):
            ax.loglog(gate_t_hm, R['fwd_hm'][j], '-', color='indianred',
                      alpha=0.20, lw=0.7)

    ax.loglog(R['raw_lm_t'], R['raw_lm_v'], 'k-',  lw=1.6, label='LM field')
    ax.loglog(R['raw_hm_t'], R['raw_hm_v'], 'k--', lw=1.6, label='HM field')
    ax.loglog(gate_t_lm, R['d_lm_med'], 'b-', lw=1.6, label='LM med→fwd')
    ax.loglog(gate_t_hm, R['d_hm_med'], 'r-', lw=1.6, label='HM med→fwd')

    if 'd_lm_O' in R:
        ax.loglog(gate_t_lm, R['d_lm_O'], 'g:', lw=1.3, alpha=0.8, label='LM Occam→fwd')
        ax.loglog(gate_t_hm, R['d_hm_O'], 'm:', lw=1.3, alpha=0.8, label='HM Occam→fwd')

    ax.set_title(f"S{s} fit={R['fit_comb']:.3f}", fontsize=8)
    ax.grid(alpha=0.3, which='both')
    if k == 0: ax.legend(fontsize=6)

for k in range(nS, nr*nc):
    axd[k // nc, k % nc].set_visible(False)
axd[-1, 0].set_xlabel("t (s)"); axd[0, 0].set_ylabel("|dB/dt|")
fig3.suptitle("Data-space fit — field vs forwarded posterior "
              "(blue/red = LM/HM samples)")
fig3.tight_layout(); fig3.savefig("cnf_datafit_grid.png", dpi=130); plt.show()

# ============================================================
# HONEST SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("HONEST SUMMARY")
print("=" * 72)
if with_occam:
    c68_top = cov68_layer[:n_top].mean()
    c95_top = cov95_layer[:n_top].mean()
    mfit = np.nanmean(med_fits); ofit = np.nanmean(occ_fits) if occ_fits else np.nan
    print(f"""
Across {len(with_occam)} soundings, top {n_top} (well-resolved) layers:
  Mean 68% coverage: {c68_top:.3f}  (target 0.68)
  Mean 95% coverage: {c95_top:.3f}  (target 0.95)
  Mean median data-fit: {mfit:.4f}   Mean Occam data-fit: {ofit:.4f}
""")
    if c68_top < 0.55 and mfit < ofit * 1.5:
        print("  VERDICT: Median fits data as well as Occam, yet coverage is low")
        print("           -> OVERCONFIDENT posterior (under-dispersed), NOT prior-")
        print("           dominated. Bands too narrow where data is informative.")
        print("           Next: run SBC on synthetic θ to confirm vs Occam bias;")
        print("           if confirmed, inflate training noise / temper the flow.")
    elif abs(c68_top - 0.68) < 0.10 and abs(c95_top - 0.95) < 0.08:
        print("  VERDICT: Calibration holds across the survey. Trustworthy inverter.")
    elif c68_top > 0.80:
        print("  VERDICT: UNDERCONFIDENT — bands too wide. Consider narrower noise.")
    else:
        print("  VERDICT: Calibration close-ish to target; some tuning could help.")
else:
    print("No Occam matches available -- calibration verdict skipped, "
          "data-fit numbers above still stand on their own.")

print("""
How to read the outputs, per sounding:
  * profile grid   -> where median/bands sit vs Occam
  * data-fit grid  -> whether those models actually reproduce the curves
  * coverage table -> whether the truth lands inside the bands
A sounding with a TIGHT data-fit spaghetti but LOW coverage is the tell-tale
overconfident case.
""")

# ============================================================
# SELECT THE 10 MOST RELIABLE SOUNDINGS
# (uses the `results` computed above; CNF input: 20 LM + 27 HM gates)
# ============================================================

import numpy as np


# ============================================================
# 1. BASIC CHECKS
# ============================================================

print("=" * 80)
print("SELECTING 10 RELIABLE SOUNDINGS")
print("=" * 80)

print("Number of evaluated soundings :", len(results))
print("LM gates                     :", len(gate_t_lm))
print("HM gates                     :", len(gate_t_hm))
print("LM normalization             :", len(xlm_mean))
print("HM normalization             :", len(xhm_mean))

assert len(gate_t_lm) == 20
assert len(gate_t_hm) == 27
assert len(xlm_mean) == 20
assert len(xhm_mean) == 27


# ============================================================
# 2. BUILD QUALITY TABLE
# ============================================================

quality = []

for s in valid:

    R = results[s]

    # --------------------------------------------------------
    # A. Fits computed in the evaluation loop above
    # --------------------------------------------------------

    fit_lm = float(R["fit_lm"])
    fit_hm = float(R["fit_hm"])
    fit_comb = float(R["fit_comb"])

    fit_samp = float(
        R.get("fit_samp_mean", np.nan)
    )

    fit_occam = float(
        R.get("fit_occam", np.nan)
    )


    # --------------------------------------------------------
    # B. Raw field gate quality
    # --------------------------------------------------------

    lm_v = np.asarray(
        soundings[s]["LM"]["v"],
        dtype=float
    )

    hm_v = np.asarray(
        soundings[s]["HM"]["v"],
        dtype=float
    )

    lm_q = np.asarray(
        soundings[s]["LM"]["q"]
    )

    hm_q = np.asarray(
        soundings[s]["HM"]["q"]
    )


    good_lm = (
        np.isfinite(lm_v) &
        (lm_v > 0) &
        (lm_q == 1)
    )

    good_hm = (
        np.isfinite(hm_v) &
        (hm_v > 0) &
        (hm_q == 1)
    )


    n_lm_good = int(
        np.sum(good_lm)
    )

    n_hm_good = int(
        np.sum(good_hm)
    )


    frac_lm = (
        n_lm_good /
        len(lm_v)
    )

    frac_hm = (
        n_hm_good /
        len(hm_v)
    )


    # --------------------------------------------------------
    # C. OOD CHECK
    #
    # IMPORTANT:
    # Use exactly the same LM/HM data that went into CNF.
    # These are already stored in results.
    # --------------------------------------------------------

    lm_data = np.asarray(
        R["lm"],
        dtype=float
    )

    hm_data = np.asarray(
        R["hm"],
        dtype=float
    )


    assert len(lm_data) == 20
    assert len(hm_data) == 27


    x_lm_log = np.log10(
        lm_data
    )

    x_hm_log = np.log10(
        hm_data
    )


    z_lm = (
        x_lm_log -
        xlm_mean
    ) / np.maximum(
        xlm_std,
        1e-8
    )


    z_hm = (
        x_hm_log -
        xhm_mean
    ) / np.maximum(
        xhm_std,
        1e-8
    )


    all_z = np.concatenate([
        z_lm,
        z_hm
    ])


    rms_z = float(
        np.sqrt(
            np.mean(all_z**2)
        )
    )


    max_z = float(
        np.max(
            np.abs(all_z)
        )
    )


    frac_z3 = float(
        np.mean(
            np.abs(all_z) <= 3
        )
    )


    # --------------------------------------------------------
    # D. Store
    # --------------------------------------------------------

    quality.append({

        "sounding": s,

        "name":
            R.get(
                "name",
                f"S{s}"
            ),

        "fit_lm":
            fit_lm,

        "fit_hm":
            fit_hm,

        "fit":
            fit_comb,

        "fit_samp":
            fit_samp,

        "occam":
            fit_occam,

        "n_lm":
            n_lm_good,

        "n_hm":
            n_hm_good,

        "frac_lm":
            frac_lm,

        "frac_hm":
            frac_hm,

        "rms_z":
            rms_z,

        "max_z":
            max_z,

        "frac_z3":
            frac_z3
    })


# ============================================================
# 3. PRINT ALL SOUNDINGS
# ============================================================

print("\n")
print("=" * 125)
print("ALL SOUNDINGS")
print("=" * 125)

print(
    f"{'S':>3} "
    f"{'LM':>7} "
    f"{'HM':>7} "
    f"{'LMfit':>8} "
    f"{'HMfit':>8} "
    f"{'Total':>8} "
    f"{'Sample':>8} "
    f"{'Occam':>8} "
    f"{'rmsZ':>7} "
    f"{'Z<=3':>7}"
)

print("-" * 125)


for x in quality:

    occ = (
        f"{x['occam']:.4f}"
        if np.isfinite(x["occam"])
        else "--"
    )

    print(
        f"{x['sounding']:3d} "
        f"{x['n_lm']:2d}/20   "
        f"{x['n_hm']:2d}/27   "
        f"{x['fit_lm']:8.4f} "
        f"{x['fit_hm']:8.4f} "
        f"{x['fit']:8.4f} "
        f"{x['fit_samp']:8.4f} "
        f"{occ:>8} "
        f"{x['rms_z']:7.2f} "
        f"{x['frac_z3']:7.2f}"
    )


# ============================================================
# 4. FIRST SELECT ONLY COMPLETE FIELD SOUNDINGS
#
# Prefer soundings with all gates present (LM = 20/20, HM = 27/27),
# because the CNF was trained on 20 LM + 27 HM gates.
# ============================================================

complete = [

    x for x in quality

    if
    x["n_lm"] == 20
    and
    x["n_hm"] == 27
]


print(
    "\nComplete 20/20 LM + 27/27 HM soundings:",
    len(complete)
)


# ============================================================
# 5. REMOVE CLEARLY BAD FORWARD FITS
#
# On this dataset the fits separate naturally:
#
# good soundings ~ 0.02 - 0.1
# bad soundings  ~ several log units
#
# 0.15 is therefore a useful screening threshold
# for THIS dataset.
# ============================================================

reliable = [

    x for x in complete

    if
    np.isfinite(x["fit"])
    and
    x["fit"] < 0.15
]


print(
    "Complete soundings with fit < 0.15:",
    len(reliable)
)


# ============================================================
# 6. RELIABILITY SCORE
#
# Primary:
#       median forward fit
#
# Secondary:
#       posterior-sample forward fit
#       distance from training distribution
#
# We are NOT using Occam agreement to select the sounding.
# ============================================================

for x in reliable:

    sample_fit = (
        x["fit_samp"]
        if np.isfinite(x["fit_samp"])
        else x["fit"]
    )

    x["score"] = (
          1.00 * x["fit"]
        + 0.20 * sample_fit
        + 0.02 * x["rms_z"]
    )


# ============================================================
# 7. SORT
# ============================================================

reliable = sorted(
    reliable,
    key=lambda x: x["score"]
)


# ============================================================
# 8. BEST 10
# ============================================================

best10 = reliable[:10]

best10_soundings = [
    x["sounding"]
    for x in best10
]


# ============================================================
# 9. PRINT FINAL 10
# ============================================================

print("\n")
print("=" * 125)
print("10 SELECTED RELIABLE SOUNDINGS")
print("=" * 125)

print(
    f"{'Rank':>5} "
    f"{'S':>4} "
    f"{'Name':<18} "
    f"{'LMfit':>8} "
    f"{'HMfit':>8} "
    f"{'Total':>8} "
    f"{'Sample':>8} "
    f"{'Occam':>8} "
    f"{'rmsZ':>7} "
    f"{'Score':>8}"
)

print("-" * 125)


for rank, x in enumerate(
    best10,
    start=1
):

    occ = (
        f"{x['occam']:.4f}"
        if np.isfinite(x["occam"])
        else "--"
    )

    print(
        f"{rank:5d} "
        f"{x['sounding']:4d} "
        f"{str(x['name']):<18} "
        f"{x['fit_lm']:8.4f} "
        f"{x['fit_hm']:8.4f} "
        f"{x['fit']:8.4f} "
        f"{x['fit_samp']:8.4f} "
        f"{occ:>8} "
        f"{x['rms_z']:7.2f} "
        f"{x['score']:8.4f}"
    )


print("\nSelected sounding numbers:")
print(best10_soundings)
