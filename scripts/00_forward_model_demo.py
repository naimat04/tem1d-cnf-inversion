"""Step 0 (optional): sanity-check the Fortran forward model on a 20-layer model and compare with a field sounding."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root
import config                                                   # all paths live in config.py
os.chdir(config.WORK_DIR)                                       # outputs (.npz/.png/checkpoints) land in outputs/

from tem1d_cnf.tem1d_wrapper import tem1d_forward

# ============================================================
# TEM1D Forward Model — LM & HM — 40×40 m loop
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# The Fortran library is loaded by tem1d_cnf.tem1d_wrapper (path set in config.py).

# =============================================================================
# INSTRUMENT CONSTANTS
# =============================================================================
TXAREA = 1600.0    # 40 m × 40 m
RTXRX  = 0.0       # Tx–Rx offset [m]

# 40×40 m square polygon (vertices at ±20 m, CCW)
xpoly = np.array([ 20.0,  20.0, -20.0, -20.0])
ypoly = np.array([ 20.0, -20.0, -20.0,  20.0])

# =============================================================================
# RESISTIVITY MODEL (20 layers)
# =============================================================================
res_data = np.array([
    [290.5,   9.049],
    [ 36.09, 19.269],
    [ 14.47, 30.799],
    [284.9,  43.819],
    [595.5,  58.519],
    [588.0,  75.119],
    [527.8,  93.859],
    [558.1, 115.019],
    [615.8, 138.909],
    [574.8, 165.879],
    [392.7, 196.319],
    [180.2, 230.689],
    [ 52.15,269.499],
    [  9.731,313.309],
    [  6.962,362.769],
    [  8.178,418.609],
    [  7.924,481.659],
    [  7.500,552.839],
    [  6.223,633.199],
    [  5.045,949.798],
])

rhon = res_data[:, 0]
depn = np.concatenate([[0.0], res_data[:-1, 1]])
nlay = len(rhon)

print(f"Model: {nlay} layers")
print(f"{'#':>3}  {'Top[m]':>10}  {'Rho[Ω·m]':>10}")
for k in range(nlay):
    print(f"{k+1:3d}  {depn[k]:10.3f}  {rhon[k]:10.3f}")

# =============================================================================
# GATE TIMES
# =============================================================================
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

# =============================================================================
# WAVEFORMS
# =============================================================================
twave_lm = np.array([
    -9.000000e-04, -8.050700e-04, -6.893900e-04, -5.855400e-04,
    -5.079800e-04, -3.686400e-04, -1.872300e-04,  0.000000e+00,
     2.070000e-07,  6.210000e-07,  8.210000e-07,  1.093000e-06,
     1.579000e-06,  2.021000e-06,  2.464000e-06,  3.121000e-06,
     4.093000e-06,  5.264000e-06,  7.000000e-06,
])
awave_lm = np.array([
    0.0000, 0.3650, 0.6110, 0.7560, 0.8230, 0.9160, 0.9700,
    1.0000, 0.9890, 0.7720, 0.7120, 0.6530, 0.4010, 0.2580,
    0.1520, 0.0750, 0.0280, 0.0110, 0.0000,
])

twave_hm = np.array([
    -6.000000e-03, -5.902800e-03, -5.691500e-03, -5.505600e-03,
    -5.277500e-03, -5.007000e-03, -4.626800e-03, -4.153500e-03,
    -2.201400e-03,  0.000000e+00,  3.450000e-07,  8.930000e-07,
     1.560000e-06,  2.107000e-06,  2.655000e-06,  3.774000e-06,
     4.845000e-06,  5.798000e-06,  6.655000e-06,  7.750000e-06,
     9.869000e-06,  1.300000e-05,
])
awave_hm = np.array([
    0.0000, 0.3060, 0.6710, 0.8290, 0.9250, 0.9710, 0.9940,
    0.9990, 1.0000, 0.9970, 0.9270, 0.8800, 0.7330, 0.6460,
    0.5300, 0.3230, 0.1500, 0.0620, 0.0300, 0.0150, 0.0060, 0.0000,
])

# =============================================================================
# COMMON FORWARD CALL KWARGS
# =============================================================================
COMMON = dict(
    txarea=TXAREA, rtxrx=RTXRX,
    ishtx1=1, ishtx2=0, ishrx1=1, ishrx2=0,
    xpoly=xpoly, ypoly=ypoly,
    x0rx=0.0, y0rx=0.0,
    iresptype=2, ideriv=0,    # ← no Jacobian — for data generation
    irep=1, iwconv=1,
    filtfreq=np.array([650000.0, 800000.0]),
)

COMMON_J = dict(
    txarea=TXAREA, rtxrx=RTXRX,
    ishtx1=1, ishtx2=0, ishrx1=1, ishrx2=0,
    xpoly=xpoly, ypoly=ypoly,
    x0rx=0.0, y0rx=0.0,
    iresptype=2, ideriv=1,    # ← analytic Jacobian ON
    irep=1, iwconv=1,
    filtfreq=np.array([650000.0, 800000.0]),
)

# Why two dicts:
# COMMON   → data generation (300k calls) — ideriv=0 is faster, no J needed
# COMMON_J → OCCAM + GN + UQ             — ideriv=1 gives exact J for free

# =============================================================================
# RUN FORWARD MODELS
# =============================================================================
print("\nRunning LM forward model…")
times_lm, resp_lm, jac_lm = tem1d_forward(
    rhon, depn, twave=twave_lm, awave=awave_lm, repfreq=512.82, **COMMON)
print(f"  LM: {len(times_lm)} gates  ({times_lm[0]:.3e} → {times_lm[-1]:.3e} s)")

print("Running HM forward model…")
times_hm, resp_hm, jac_hm = tem1d_forward(
    rhon, depn, twave=twave_hm, awave=awave_hm, repfreq=50.00, **COMMON)
print(f"  HM: {len(times_hm)} gates  ({times_hm[0]:.3e} → {times_hm[-1]:.3e} s)")


# =============================================================================
# INTERPOLATE ONTO GATE TIMES (log-log)
# =============================================================================
SCALE = TXAREA  # 1600.0

def interp_loglog(t_model, r_model, t_query):
    lt = np.log10(t_model)
    lr = np.log10(np.abs(r_model))
    lt_q = np.clip(np.log10(t_query), lt[0], lt[-1])
    return 10.0 ** np.interp(lt_q, lt, lr)

pred_lm = SCALE * interp_loglog(times_lm, resp_lm, gate_t_lm)
pred_hm = SCALE * interp_loglog(times_hm, resp_hm, gate_t_hm)

valid_lm = (gate_t_lm >= times_lm[0]) & (gate_t_lm <= times_lm[-1])
valid_hm = (gate_t_hm >= times_hm[0]) & (gate_t_hm <= times_hm[-1])

print(f"\nLM: {valid_lm.sum()}/{len(gate_t_lm)} gates in range")
print(f"HM: {valid_hm.sum()}/{len(gate_t_hm)} gates in range")

# =============================================================================
# APPARENT RESISTIVITY
# =============================================================================
MU0    = 4 * np.pi * 1e-7
PREFAC = MU0 ** (5/3) / (np.pi ** (1/3))

def apparent_rho(t, dbdt, A=TXAREA, I=1.0):
    rho = np.full_like(t, np.nan)
    m   = dbdt > 0
    if m.any():
        rho[m] = ((I * A**2) / (20.0 * dbdt[m])) ** (2/3) * PREFAC * t[m] ** (-5/3)
    return rho

rho_lm = apparent_rho(gate_t_lm[valid_lm], pred_lm[valid_lm])
rho_hm = apparent_rho(gate_t_hm[valid_hm], pred_hm[valid_hm])

# =============================================================================
# LOAD FIELD DATA (optional — LM40.csv / HM40.csv in the data folder)
# =============================================================================
DATASET = str(config.DATA_DIR)

def load_csv(fname):
    p = Path(DATASET) / fname
    if not p.exists():
        print(f"  ⚠ {fname} not found — skipping field data")
        return None, None
    with open(p, encoding='utf-8-sig') as f:
        d = np.loadtxt(f, delimiter=',')
    return (d[:, 0], d[:, 1]) if d.ndim == 2 else (None, None)

field_t_lm, field_b_lm = load_csv("LM40.csv")
field_t_hm, field_b_hm = load_csv("HM40.csv")
have_field = field_t_lm is not None and field_t_hm is not None

if have_field:
    rho_field_lm = apparent_rho(field_t_lm, np.abs(field_b_lm))
    rho_field_hm = apparent_rho(field_t_hm, np.abs(field_b_hm))
    print("✓ Field data loaded")

# =============================================================================
# SUMMARY TABLES
# =============================================================================
# for label, gate_t, pred, valid, rho_app in [
#     ("LM", gate_t_lm, pred_lm, valid_lm, apparent_rho(gate_t_lm, pred_lm)),
#     ("HM", gate_t_hm, pred_hm, valid_hm, apparent_rho(gate_t_hm, pred_hm)),
# ]:
#     print(f"\n── {label} gate predictions ──────────────────────────────")
#     print(f"  {'#':>2}  {'Time [s]':>12}  {'dB/dt [V/m²]':>14}  {'ρa [Ω·m]':>12}  In range")
#     for i, t in enumerate(gate_t):
#         r = "yes" if valid[i] else "no"
#         rho_str = f"{rho_app[i]:12.4e}" if valid[i] else "           —"
#         print(f"  {i+1:2d}  {t:.6e}  {pred[i]:14.6e}  {rho_str}  {r}")

# =============================================================================
# PLOTS
# =============================================================================
def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')

# ── Fig 1 : dB/dt raw curves + gate predictions ─────────────
fig1, ax1 = plt.subplots(figsize=(11, 7))
ax1.loglog(times_hm, SCALE*np.abs(resp_hm), 'b:', lw=1, label="Model HM (raw)")
ax1.loglog(times_lm, SCALE*np.abs(resp_lm), 'r:', lw=1, label="Model LM (raw)")
ax1.loglog(gate_t_hm[valid_hm], pred_hm[valid_hm], 'bo', ms=6, label="Pred HM (gates)")
ax1.loglog(gate_t_lm[valid_lm], pred_lm[valid_lm], 'ro', ms=6, label="Pred LM (gates)")
if have_field:
    ax1.loglog(field_t_lm, np.abs(field_b_lm), 'r-', lw=2, label="Field LM")
    ax1.loglog(field_t_hm, np.abs(field_b_hm), 'b-', lw=2, label="Field HM")
style(ax1, "TEM1D dB/dt — LM & HM (40×40 m loop)", "Time (s)", "|dB/dt| (V/m²)")
fig1.tight_layout(); fig1.savefig("tem1d_lm_hm.png", dpi=150)

# ── Fig 2 : Apparent resistivity ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 7))
if have_field:
    ax2.loglog(field_t_lm, rho_field_lm, 'r-', lw=2, label="Field LM")
    ax2.loglog(field_t_hm, rho_field_hm, 'b-', lw=2, label="Field HM")
ax2.loglog(gate_t_lm[valid_lm], rho_lm, 'ro', ms=6, label="Model LM")
ax2.loglog(gate_t_hm[valid_hm], rho_hm, 'bo', ms=6, label="Model HM")
style(ax2, "Apparent Resistivity — LM & HM (40×40 m loop)", "Time (s)", "ρₐ (Ω·m)")
fig2.tight_layout(); fig2.savefig("tem1d_apparent_rho.png", dpi=150)

# ── Fig 3 : Gate predictions vs field ───────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 7))
ax3.loglog(gate_t_hm[valid_hm], pred_hm[valid_hm], 'bo', ms=7, label="Predicted HM")
ax3.loglog(gate_t_lm[valid_lm], pred_lm[valid_lm], 'ro', ms=7, label="Predicted LM")
if have_field:
    ax3.loglog(field_t_lm, np.abs(field_b_lm), 'r-', lw=2, label="Field LM")
    ax3.loglog(field_t_hm, np.abs(field_b_hm), 'b-', lw=2, label="Field HM")
style(ax3, "Model (gate-interpolated) vs Field — LM & HM", "Time (s)", "|dB/dt| (V/m²)")
fig3.tight_layout(); fig3.savefig("tem1d_interp_vs_field.png", dpi=150)

# ── Fig 4 : ρa model vs field ───────────────────────────────
fig4, ax4 = plt.subplots(figsize=(11, 7))
if have_field:
    ax4.loglog(field_t_lm, rho_field_lm, 'r-', lw=2, label="Field LM")
    ax4.loglog(field_t_hm, rho_field_hm, 'b-', lw=2, label="Field HM")
ax4.loglog(gate_t_lm[valid_lm], rho_lm, 'ro', ms=7, label="Model LM (gates)")
ax4.loglog(gate_t_hm[valid_hm], rho_hm, 'bo', ms=7, label="Model HM (gates)")
style(ax4, "Apparent Resistivity: Model vs Field — LM & HM", "Time (s)", "ρₐ (Ω·m)")
fig4.tight_layout(); fig4.savefig("tem1d_apparent_rho_vs_field.png", dpi=150)

plt.show()
print("\n✓ All plots saved")
