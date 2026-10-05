# Conditional normalizing flows for Bayesian 1D TEM inversion — code

Code accompanying the manuscript *"Conditional Normalizing Flows for Bayesian Inversion of Transient
Electromagnetic Data"*. A conditional normalizing flow (zuko Neural Spline Flow, conditioned on a two-branch
ResNet encoder of the low-moment (LM) and high-moment (HM) decays) is trained on synthetic TEM1D data and
then applied to field soundings to obtain full posterior resistivity profiles.

## Layout

```
config.py                         all paths (override with TEM_DATA_DIR / LIBTEM1D / TEM_WORK_DIR)
tem1d_cnf/
  tem1d_wrapper.py                ctypes wrapper around the TEM1D Fortran library
  instrument.py                   loop geometry, LM/HM gate times, current waveforms
  forward.py                      forward model on the instrument gates (+ Jacobian)
scripts/
  00_forward_model_demo.py        (optional) forward-model sanity check vs a field sounding
  01_generate_training_data.py    synthetic (model, LM+HM data) pairs from simple smooth random resistivity models
  02_train_cnf.py                 train encoder + flow, calibration / rank-histogram plots
  03_evaluate_single_sounding.py  posterior for one field sounding (68 % / 95 % bands, forward consistency)
  04_evaluate_all_soundings.py    all 8 soundings, pseudo-sections, data-fit grids, reliable-sounding selection
fortran/                          where to put libtem1d.so (build instructions for TEM1D, not bundled)
data/  outputs/                   field data go in data/ (not distributed); results are written to outputs/
```

## Setup

```bash
pip install -r requirements.txt     # a CUDA build of PyTorch is recommended for steps 01-02
```

Build the TEM1D Fortran library from its original repository and place `libtem1d.so` in `fortran/` (instructions in `fortran/README.md`; needs `gfortran`).
The field data are **not distributed** with this code; `data/README.md` lists the expected files and formats.

## Run order

```bash
python scripts/00_forward_model_demo.py          # optional
python scripts/01_generate_training_data.py      # -> outputs/tem1d_train.npz
python scripts/02_train_cnf.py                   # -> outputs/nn_cnf_1/{encoder_best.pt, flow_best.pt, norm_stats.npz, ...}
python scripts/03_evaluate_single_sounding.py    # -> outputs/cnf_field_posterior.png
python scripts/04_evaluate_all_soundings.py      # -> outputs/cnf_sections.png, cnf_soundings_grid.png, cnf_datafit_grid.png
```

Set `MPLBACKEND=Agg` on a machine without a display; figures are saved either way.

## Reproducibility notes

- Training seed is fixed (`SEED = 42` in `02_train_cnf.py`); training noise is ~3 % multiplicative log-Gaussian.
- Step 02 stores the number of dropped trailing LM gates in `norm_stats.npz`; step 04 reads it back and
  asserts consistency, so training and inference cannot silently drift apart.
- Depth grid: 18 defined layers + half-space (≈ 312 m), reproducing the real Aarhus SPIA grid for layers 1–18.

## Third-party code

The forward modelling relies on the open-source TEM1D FORTRAN subroutine (Christensen et al., 2026,
*Computers & Geosciences* 209, 106102, https://doi.org/10.1016/j.cageo.2025.106102;
source: https://github.com/hydrogeophysicsgroup/TEM1D). It is not redistributed here; see `fortran/README.md`.

## Citation / licence

TODO (author): add citation of the manuscript, DOI of the archived release, and a licence for this code.
