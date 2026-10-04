import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
import config                                                   # all paths live in config.py
os.chdir(config.WORK_DIR)                                       # outputs are written to outputs/

import time
import numpy as np
from tem1d_cnf.tem1d_wrapper import tem1d_forward

# ============================================================
#  Synthetic training data for the CNF inverter
#
#  Each training example is a pair
#       (resistivity model)  ->  (LM + HM transient data)
#  The data are computed by forward modelling with TEM1D.
#
#  Steps
#    1. Fixed depth grid     : 18 layers + half-space (0 -> ~312 m)
#    2. Instrument           : loop, gate times and waveforms (LM and HM)
#    3. Random model         : smooth random log10(resistivity) profile
#    4. Depth of investigation (DOI) and per-layer weight (diagnostics)
#    5. Forward model        : TEM1D response sampled at the gate times
#    6. Generation loop      : repeat until n_samples valid pairs are saved
#
#  To change the kind of ground the network is trained on, edit only
#  the settings in section 3 (RESISTIVITY RANGE and MODEL SHAPE).
#
#  Output: outputs/tem1d_train.npz   (read by 02_train_cnf.py)
#  Needs the compiled TEM1D library (see fortran/README.md).
# ============================================================

# =============================================================================
# 1. FIXED DEPTH GRID
# =============================================================================
#  Layer thicknesses grow geometrically with depth (thin near the surface,
#  thick at depth). The last layer is a half-space whose thickness is unused.
N_LAY   = 19           # 18 finite layers + 1 half-space
THK_TOP = 5.10622      # thickness of the first layer (m)
THK_BOT = 40.165575    # thickness of the 18th layer (m)

thk  = np.geomspace(THK_TOP, THK_BOT, N_LAY - 1)        # 18 finite layers
thk  = np.append(thk, thk[-1])                          # half-space
depn = np.concatenate([[0.0], np.cumsum(thk)[:-1]])     # depth to the TOP of each layer
zc   = depn + thk / 2.0                                 # depth of each layer centre
print(f"Depth grid: 0 -> {depn[-1]:.1f} m (top of half-space),  {N_LAY} layers")

# =============================================================================
# 2. INSTRUMENT / WAVEFORM  (same definitions as tem1d_cnf/instrument.py)
# =============================================================================
TXAREA = 1600.0
xpoly  = np.array([ 20.0,  20.0, -20.0, -20.0])
ypoly  = np.array([ 20.0, -20.0, -20.0,  20.0])

gate_t_lm = np.array([
    1.24548e-05, 1.55122e-05, 1.93201e-05, 2.40627e-05, 2.99696e-05,
    3.73264e-05, 4.64893e-05, 5.79014e-05, 7.21149e-05, 8.98175e-05,
    1.11866e-04, 1.39326e-04, 1.73528e-04, 2.16125e-04, 2.69179e-04,
    3.35257e-04, 4.17555e-04, 5.20056e-04, 6.47718e-04, 8.06718e-04,
])
gate_t_hm = np.array([
    3.13353e-05, 3.92761e-05, 4.92291e-05, 6.17044e-05, 7.73411e-05,
    9.69403e-05, 1.21506e-04, 1.52297e-04, 1.90892e-04, 2.39266e-04,
    2.99899e-04, 3.75897e-04, 4.71154e-04, 5.90551e-04, 7.40204e-04,
    9.27781e-04, 1.16289e-03, 1.45758e-03, 1.82695e-03, 2.28993e-03,
    2.87022e-03, 3.59757e-03, 4.50924e-03, 5.65194e-03, 7.08422e-03,
    8.87945e-03, 1.11296e-02,
])
N_LM, N_HM = len(gate_t_lm), len(gate_t_hm)

twave_lm = np.array([-9.0000e-04,-8.0507e-04,-6.8939e-04,-5.8554e-04,-5.0798e-04,
                     -3.6864e-04,-1.8723e-04, 0.0000e+00, 2.070e-07, 6.210e-07,
                      8.210e-07, 1.093e-06, 1.579e-06, 2.021e-06, 2.464e-06,
                      3.121e-06, 4.093e-06, 5.264e-06, 7.000e-06])
awave_lm = np.array([0.000,0.365,0.611,0.756,0.823,0.916,0.970,1.000,0.989,
                     0.772,0.712,0.653,0.401,0.258,0.152,0.075,0.028,0.011,0.000])

twave_hm = np.array([-6.0000e-03,-5.9028e-03,-5.6915e-03,-5.5056e-03,-5.2775e-03,
                     -5.0070e-03,-4.6268e-03,-4.1535e-03,-2.2014e-03, 0.0000e+00,
                      3.450e-07, 8.930e-07, 1.560e-06, 2.107e-06, 2.655e-06,
                      3.774e-06, 4.845e-06, 5.798e-06, 6.655e-06, 7.750e-06,
                      9.869e-06, 1.300e-05])
awave_hm = np.array([0.000,0.306,0.671,0.829,0.925,0.971,0.994,0.999,1.000,0.997,
                     0.927,0.880,0.733,0.646,0.530,0.323,0.150,0.062,0.030,0.015,
                     0.006,0.000])

FWD_COMMON = dict(
    txarea=TXAREA, rtxrx=0.0,
    ishtx1=1, ishtx2=0, ishrx1=1, ishrx2=0,
    xpoly=xpoly, ypoly=ypoly, x0rx=0.0, y0rx=0.0,
    iresptype=2, ideriv=0, irep=1, iwconv=1,     # ideriv=0: no Jacobian needed, faster
    filtfreq=np.array([650000.0, 800000.0]),
)

# =============================================================================
# 3. RANDOM RESISTIVITY MODEL
# =============================================================================
#  Every model is a smooth random profile of log10(resistivity) over the
#  N_LAY layers of the fixed grid:
#
#     log10(rho) = base level + smooth random variation with depth
#                  (+ optionally a conductive lens)
#                  (+ optionally a resistive cap at the surface)
#
#  "Smooth" means neighbouring layers have similar values; the random
#  variation changes gradually over ~MODEL_SMOOTHNESS layers.

# ---- RESISTIVITY RANGE ----
LOG_RHO_MIN, LOG_RHO_MAX = -1.0, 3.5      # hard limits: 0.1 to ~3162 ohm.m

# ---- MODEL SHAPE ----
BASE_LEVEL      = (0.9, 1.9)    # average log10(rho) of a model, drawn uniformly (8 - 80 ohm.m)
VARIATION       = (0.30, 0.55)  # strength of the variation with depth (log10 units)
MODEL_SMOOTHNESS = 2.0          # correlation length in layers (larger = smoother)
LENS_PROB       = 0.25          # chance of adding a conductive lens (e.g. clay or saline water)
CAP_PROB        = 0.15          # chance of adding a resistive layer at the surface (e.g. dry soil)

# Smoothing matrix: correlated random numbers are produced by multiplying
# independent random numbers with this matrix.
_layer = np.arange(N_LAY, dtype=float)
_corr  = np.exp(-0.5 * (_layer[:, None] - _layer[None, :]) ** 2 / MODEL_SMOOTHNESS ** 2)
_smooth = np.linalg.cholesky(_corr + 1e-6 * np.eye(N_LAY))


def sample_prior_model(rng):
    """Draw one random resistivity model. Returns rho (ohm.m), shape (N_LAY,)."""
    # base level + smooth variation with depth
    log_rho = rng.uniform(*BASE_LEVEL) \
              + rng.uniform(*VARIATION) * (_smooth @ rng.standard_normal(N_LAY))

    # conductive lens: a smooth dip at a random depth
    if rng.random() < LENS_PROB:
        centre = rng.uniform(3, N_LAY - 3)       # position (layer index)
        width  = rng.uniform(1.0, 2.5)           # width (layers)
        depth_of_dip = rng.uniform(0.4, 1.2)     # strength (log10 units)
        log_rho -= depth_of_dip * np.exp(-0.5 * ((_layer - centre) / width) ** 2)

    # resistive cap: a smooth rise at the surface
    if rng.random() < CAP_PROB:
        log_rho += rng.uniform(0.4, 1.0) * np.exp(-0.5 * (_layer / 1.2) ** 2)

    return 10.0 ** np.clip(log_rho, LOG_RHO_MIN, LOG_RHO_MAX)

# =============================================================================
# 4. DEPTH OF INVESTIGATION (DOI) AND LAYER WEIGHT
# =============================================================================
#  How deep the data can "see" depends on how conductive the ground above is:
#  a conductive near-surface layer screens the deeper structure. The DOI is
#  therefore estimated separately for every model with the Spies (1989)
#  formula
#       D = 528 * sqrt(rho_app * t)      [rho_app: ohm.m, t: s, D: m]
#  where rho_app is the depth-averaged apparent resistivity above depth z and
#  t is the last HM gate time.
#
#  Each model is stored together with its DOI and a weight per layer
#  (about 1 above the DOI, falling to 0 below it). They are saved for
#  diagnostics only and are not used to change the models.
# =============================================================================
T_DOI = float(gate_t_hm[-1])     # last HM gate time (deepest sensing)
DOI_CONST = 528.0                # Spies (1989) constant
DOI_TAPER_DEX = 0.3              # width of the weight roll-off (decades of depth)


def apparent_resistivity_cumulative(rhon, thk_arr, dep_arr):
    """Dar-Zarrouk depth-averaged apparent resistivity down to the bottom of
    each finite layer. rhon/thk_arr must be the FINITE layers only (no
    half-space); dep_arr is the depth-to-TOP array (same length as rhon,
    starting at 0), matching this script's `depn` convention.
    Returns (z, rho_app) both length = len(rhon)."""
    rhon = np.asarray(rhon, float)
    thk_arr = np.asarray(thk_arr, float)
    S = np.cumsum(thk_arr / rhon)                 # cumulative conductance
    z = dep_arr[1:] if len(dep_arr) > len(rhon) else np.cumsum(thk_arr)
    z = np.asarray(z, float)
    S = np.where(S <= 0, 1e-9, S)
    return z, z / S


def model_doi(rhon, thk_arr, dep_arr, t=T_DOI, const=DOI_CONST):
    """Self-consistent DOI (m) for one model: largest z where the Spies depth
    estimate (using apparent rho ABOVE z) still reaches z."""
    z, rho_app = apparent_resistivity_cumulative(rhon, thk_arr, dep_arr)
    d_est = const * np.sqrt(np.clip(rho_app, 1e-3, None) * t)
    ok = d_est >= z
    return float(z[ok][-1]) if np.any(ok) else float(z[0])


def layer_confidence_weight(rhon_fixed_grid, t=T_DOI, const=DOI_CONST,
                             taper_dex=DOI_TAPER_DEX):
    """Per-layer weight (N_LAY,) in [0, 1]: about 1 above the model's DOI,
    falling smoothly to 0 below it. Returns (weights, doi)."""
    doi = model_doi(rhon_fixed_grid[:-1], thk[:-1], depn[:len(rhon_fixed_grid)])
    log_ratio = np.log10(np.clip(zc, 1e-3, None) / doi)
    w = 1.0 / (1.0 + 10 ** (log_ratio / taper_dex))
    return w.astype(np.float32), doi

# =============================================================================
# 5. FORWARD MODEL  (LM and HM data at the gate times)
# =============================================================================
def interp_loglog(t_model, r_model, t_query):
    lt = np.log10(t_model)
    lr = np.log10(np.abs(r_model))
    lt_q = np.clip(np.log10(t_query), lt[0], lt[-1])
    return 10.0 ** np.interp(lt_q, lt, lr)


def forward_one(rhon):
    """Run TEM1D for one model and interpolate the response onto the LM and HM
    gate times. Returns (d_lm, d_hm), or None if the response is unusable."""
    try:
        # low moment
        t_m, r_m, _ = tem1d_forward(
            rhon, depn, twave=twave_lm, awave=awave_lm,
            repfreq=512.82, **FWD_COMMON)
        r_m = r_m[:len(t_m)]
        flips = np.where(np.diff(np.sign(r_m)) != 0)[0]
        start = flips.max() + 1 if flips.size else 0
        t_m, r_m = t_m[start:], r_m[start:]
        if np.any(np.diff(np.sign(r_m)) != 0): return None
        if len(t_m) < 3 or t_m[-1] < gate_t_lm[-1] or t_m[0] > gate_t_lm[0]:
            return None
        d_lm = TXAREA * interp_loglog(t_m, r_m, gate_t_lm)

        # high moment
        t_m, r_m, _ = tem1d_forward(
            rhon, depn, twave=twave_hm, awave=awave_hm,
            repfreq=50.0, **FWD_COMMON)
        r_m = r_m[:len(t_m)]
        flips = np.where(np.diff(np.sign(r_m)) != 0)[0]
        start = flips.max() + 1 if flips.size else 0
        t_m, r_m = t_m[start:], r_m[start:]
        if np.any(np.diff(np.sign(r_m)) != 0): return None
        if len(t_m) < 3 or t_m[-1] < gate_t_hm[-1] or t_m[0] > gate_t_hm[0]:
            return None
        d_hm = TXAREA * interp_loglog(t_m, r_m, gate_t_hm)

        if not (np.all(np.isfinite(d_lm)) and np.all(np.isfinite(d_hm))):
            return None
        if np.any(d_lm <= 0) or np.any(d_hm <= 0):
            return None
        return d_lm, d_hm
    except Exception:
        return None

# =============================================================================
# 6. GENERATION LOOP
# =============================================================================
def generate_dataset(n_samples, seed=0, out_path="tem1d_train.npz",
                     verbose_every=500, heartbeat_sec=5.0, checkpoint_every=1000):
    """Generate n_samples (model, data) pairs and save them to out_path (.npz).
    Models whose forward response is unusable are skipped and redrawn."""
    rng = np.random.default_rng(seed)

    M   = np.zeros((n_samples, N_LAY), dtype=np.float32)   # log10 resistivity models
    DLM = np.zeros((n_samples, N_LM),  dtype=np.float32)   # low-moment data
    DHM = np.zeros((n_samples, N_HM),  dtype=np.float32)   # high-moment data
    W   = np.zeros((n_samples, N_LAY), dtype=np.float32)   # layer weights
    DOI = np.zeros((n_samples,),       dtype=np.float32)   # depth of investigation (m)

    n_ok, n_try = 0, 0
    t0 = time.time(); t_last_print = t0; last_checkpoint = 0

    print(f"Starting generation: target = {n_samples} samples", flush=True)

    def _save(path, k):
        np.savez_compressed(
            path,
            m_log10_rho  = M[:k],
            d_lm         = DLM[:k],
            d_hm         = DHM[:k],
            layer_weight = W[:k],
            doi          = DOI[:k],
            depn         = depn.astype(np.float32),
            thk          = thk.astype(np.float32),
            zc           = zc.astype(np.float32),
            gate_t_lm    = gate_t_lm.astype(np.float32),
            gate_t_hm    = gate_t_hm.astype(np.float32),
            doi_const    = np.float32(DOI_CONST),
            doi_t        = np.float32(T_DOI),
        )

    while n_ok < n_samples:
        n_try += 1
        rhon = sample_prior_model(rng)
        out = forward_one(rhon)
        if out is not None:                      # keep only valid forward responses
            d_lm, d_hm = out
            w, doi = layer_confidence_weight(rhon)
            M[n_ok]   = np.log10(rhon)
            DLM[n_ok] = d_lm
            DHM[n_ok] = d_hm
            W[n_ok]   = w
            DOI[n_ok] = doi
            n_ok += 1

        now = time.time()
        if (n_ok > 0 and n_ok % verbose_every == 0) \
           or (now - t_last_print > heartbeat_sec) or (n_ok == n_samples):
            elapsed = now - t0
            rate = n_ok / max(elapsed, 1e-9)
            eta  = (n_samples - n_ok) / max(rate, 1e-9)
            rej  = 1.0 - n_ok / max(n_try, 1)
            print(f"  ok {n_ok:6d}/{n_samples}   tries {n_try:7d}   "
                  f"reject {rej*100:5.1f}%   {rate:5.2f} samp/s   "
                  f"ETA {eta/60:6.1f} min", flush=True)
            t_last_print = now

        if checkpoint_every > 0 and n_ok - last_checkpoint >= checkpoint_every:
            ckpt = out_path.replace(".npz", ".ckpt.npz")
            _save(ckpt, n_ok)                    # periodic backup in case of interruption
            last_checkpoint = n_ok
            print(f"  [checkpoint] saved {n_ok} samples -> {ckpt}", flush=True)

    _save(out_path, n_ok)
    print(f"\nSaved {n_ok} samples to {out_path}", flush=True)
    print(f"  DOI over dataset -> mean:{DOI[:n_ok].mean():.1f} m  "
          f"median:{np.median(DOI[:n_ok]):.1f} m  "
          f"[{DOI[:n_ok].min():.1f}, {DOI[:n_ok].max():.1f}] m", flush=True)
    print(f"Total tries: {n_try}   overall reject: {(1-n_ok/max(n_try,1))*100:.1f}%",
          flush=True)
    return out_path

# =============================================================================
# 7. VISUAL CHECK
# =============================================================================
def preview(n=8, seed=1):
    """Plot n random models (top), their layer weights (middle) and their
    LM / HM data (bottom). Saved as training_data_preview.png."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, n, figsize=(3*n, 11))
    good = 0
    z_top = depn
    z_bot = np.concatenate([depn[1:], [depn[-1] + thk[-1]*2]])
    while good < n:
        rhon = sample_prior_model(rng)
        out = forward_one(rhon)
        if out is None:
            continue
        d_lm, d_hm = out
        w, doi = layer_confidence_weight(rhon)

        ax = axes[0, good]
        for i in range(N_LAY):
            ax.plot([rhon[i], rhon[i]], [z_top[i], z_bot[i]], 'r-', lw=2)
        for i in range(N_LAY - 1):
            ax.plot([rhon[i], rhon[i+1]], [z_bot[i], z_bot[i]], 'r-', lw=2)
        ax.axhline(doi, color='k', ls='--', lw=1.2, label=f'DOI={doi:.0f}m')
        ax.set_xscale('log'); ax.invert_yaxis()
        ax.set_xlim(10**LOG_RHO_MIN, 10**LOG_RHO_MAX)
        ax.set_xlabel('rho (ohm.m)'); ax.set_ylabel('depth (m)')
        ax.set_title(f'sample {good}'); ax.grid(alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1, good]
        ax.plot(w, zc, 'g-o', ms=3)
        ax.invert_yaxis(); ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('layer weight'); ax.set_ylabel('depth (m)')
        ax.grid(alpha=0.3)

        ax = axes[2, good]
        ax.loglog(gate_t_lm, d_lm, 'ro-', ms=4, label='LM')
        ax.loglog(gate_t_hm, d_hm, 'bo-', ms=4, label='HM')
        ax.set_xlabel('t (s)'); ax.set_ylabel('|dB/dt|')
        ax.grid(alpha=0.3, which='both')
        if good == 0: ax.legend()
        good += 1
    fig.suptitle('Random training models (row 1), layer weights (row 2) and LM/HM data (row 3)')
    fig.tight_layout()
    fig.savefig('training_data_preview.png', dpi=120)
    plt.show()
    print("Saved training_data_preview.png")


def check_prior_quantiles(n_samples=20000, seed=123):
    """Draw many models (no forward modelling, fast) and print the quantiles
    of their resistivities, to check that the range looks sensible."""
    rng = np.random.default_rng(seed)
    all_rho = np.concatenate([sample_prior_model(rng) for _ in range(n_samples)])
    qs = [0, 5, 25, 50, 75, 95, 99, 100]
    print("Resistivity quantiles of the random models (ohm.m):")
    for q, v in zip(qs, np.percentile(all_rho, qs)):
        print(f"  p{q:3d}: {v:8.3f}")
    clip_lo = (np.log10(all_rho) <= LOG_RHO_MIN + 1e-6).mean() * 100
    clip_hi = (np.log10(all_rho) >= LOG_RHO_MAX - 1e-6).mean() * 100
    print(f"  clipped at lower limit ({10**LOG_RHO_MIN} ohm.m): {clip_lo:.2f}%")
    print(f"  clipped at upper limit ({10**LOG_RHO_MAX:.0f} ohm.m): {clip_hi:.2f}%")
    return all_rho

# =============================================================================
# 8. RUN
# =============================================================================
if __name__ == "__main__":
    check_prior_quantiles()        # quick check of the resistivity range
    preview(n=8)                   # plots a few models and their data
    generate_dataset(
        n_samples = 50000,         # number of training examples
        seed      = 42,
        out_path  = str(config.TRAIN_NPZ),
    )
