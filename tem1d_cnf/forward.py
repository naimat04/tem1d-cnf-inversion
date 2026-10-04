"""Forward modelling on the instrument gates: TEM1D call + log-log interpolation (+ Jacobian)."""
import numpy as np

from tem1d_cnf.tem1d_wrapper import tem1d_forward
from tem1d_cnf.instrument import TXAREA, COMMON, COMMON_J

def _mask_before_flip(t_model, r_model, t_min_keep):
    """Keep model samples that are (a) monotonic in sign after t_min_keep and
    (b) above the earliest useful time. Guards against on-time / off-time
    sign transitions polluting the log interpolation."""
    n = len(t_model)
    r = r_model[:n]
    flips = np.where(np.diff(np.sign(r)) != 0)[0]
    if flips.size:
        last_flip = flips.max()
        # keep everything strictly AFTER the last flip
        start = last_flip + 1
    else:
        start = 0
    # also enforce t >= t_min_keep so we never query outside model range
    start = max(start, int(np.searchsorted(t_model, t_min_keep)))
    return start

def interp_resp_and_jac(t_model, r_model, J_model, t_query, scale=1.0):
    """
    Log-log interpolate response onto t_query and consistently propagate
    the Jacobian.
    Returns
        pred   : (n_gates,)        SCALE * |resp| interpolated
        J_pred : (n_gates, n_par)  d(pred)/d(sigma) at gate times
        valid  : (n_gates,) bool   True where gate lies inside model range
    """
    lt = np.log10(t_model)
    lr = np.log10(np.abs(r_model))

    valid = (t_query >= t_model[0]) & (t_query <= t_model[-1])
    lt_q  = np.clip(np.log10(t_query), lt[0], lt[-1])

    k = np.searchsorted(lt, lt_q) - 1
    k = np.clip(k, 0, len(lt) - 2)
    w = (lt_q - lt[k]) / (lt[k+1] - lt[k])

    lr_q = (1 - w) * lr[k] + w * lr[k+1]
    pred = scale * 10.0 ** lr_q

    inv_rk  = 1.0 / r_model[k]
    inv_rk1 = 1.0 / r_model[k+1]
    J_pred = pred[:, None] * (
        (1 - w)[:, None] * inv_rk [:, None] * J_model[k,   :] +
             w  [:, None] * inv_rk1[:, None] * J_model[k+1, :]
    )
    # Zero out anything extrapolated — carries no information
    J_pred[~valid, :] = 0.0
    return pred, J_pred, valid

def forward_on_gates(rhon, depn, twave, awave, repfreq, gate_t,
                     want_jac=False, scale=TXAREA):
    """Full pipeline: run TEM1D, filter, interpolate onto gate_t, and
    (optionally) propagate the Jacobian.
    Returns pred, J_pred (or None), valid mask."""
    kw = COMMON_J if want_jac else COMMON
    t_m, r_m, J_m = tem1d_forward(
        rhon, depn, twave=twave, awave=awave, repfreq=repfreq, **kw)
    n = len(t_m)
    r_m = r_m[:n]

    # Filter early-time sign flip (on-time/off-time transition) if any.
    t_min_keep = gate_t.min() * 0.5
    start = _mask_before_flip(t_m, r_m, t_min_keep)
    t_use, r_use = t_m[start:], r_m[start:]

    # Sanity: no residual sign flips in the useful window
    residual = np.sum(np.diff(np.sign(r_use)) != 0)
    assert residual == 0, f"Sign flip inside gate range at start={start}"

    if want_jac:
        nparm = len(rhon) + 1                          # MLM: sigmas + dHtx
        J_use = J_m[start:n, :nparm]
        pred, J_pred, valid = interp_resp_and_jac(
            t_use, r_use, J_use, gate_t, scale=scale)
        return pred, J_pred, valid
    else:
        # response-only path (same interpolation as interp_loglog in 00_forward_model_demo.py)
        lt = np.log10(t_use); lr = np.log10(np.abs(r_use))
        lt_q = np.clip(np.log10(gate_t), lt[0], lt[-1])
        pred = scale * 10.0 ** np.interp(lt_q, lt, lr)
        valid = (gate_t >= t_use[0]) & (gate_t <= t_use[-1])
        return pred, None, valid
