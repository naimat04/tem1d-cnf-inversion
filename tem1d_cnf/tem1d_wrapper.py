"""ctypes wrapper around the compiled TEM1D Fortran subroutine (libtem1d.so)."""
import ctypes
import numpy as np
from pathlib import Path

import config

# Fixed sizes matching Fortran PARAMETER statements
N_ARR  = 512
N_FILT = 16

# Load the library
LIB_PATH = str(config.LIB_PATH)
lib = ctypes.CDLL(LIB_PATH)
print("✓ Loaded:", lib)

# Define the Fortran subroutine signature
tem1d_ = lib.tem1d_
tem1d_.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int32),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int32),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int32),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='F_CONTIGUOUS'),
]
tem1d_.restype = None

def _pad_to_n(arr, n=N_ARR):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) >= n:
        return np.ascontiguousarray(arr[:n])
    return np.ascontiguousarray(np.concatenate([arr, np.zeros(n - len(arr))]))

def tem1d_forward(rhon, depn, *, twave, awave,
    imlm=1, txarea=1600.0, rtxrx=13.0, izeropos=0,
    ishtx1=1, ishtx2=0, ishrx1=1, ishrx2=0,
    htx1=0.0, htx2=0.0, hrx1=0.0, hrx2=0.0,
    iresptype=2, ideriv=1, irep=1, iwconv=1,
    repfreq=217.39, filtfreq=np.array([450000.0, 800000.0]),
    imodip=0, chaip=None, tauip=None, powip=None,
    xpoly=np.array([20.0, 20.0, -20.0, -20.0]),
    ypoly=np.array([20.0, -20.0, -20.0, 20.0]),
    x0rx=0.0, y0rx=0.0):

    rhon     = np.asarray(rhon,     dtype=np.float64)
    depn     = np.asarray(depn,     dtype=np.float64)
    twave    = np.asarray(twave,    dtype=np.float64)
    awave    = np.asarray(awave,    dtype=np.float64)
    filtfreq = np.asarray(filtfreq, dtype=np.float64)
    xpoly    = np.asarray(xpoly,    dtype=np.float64)
    ypoly    = np.asarray(ypoly,    dtype=np.float64)
    nlay     = len(rhon)

    chaip = np.zeros(nlay, dtype=np.float64) if chaip is None else np.asarray(chaip, dtype=np.float64)
    tauip = np.zeros(nlay, dtype=np.float64) if tauip is None else np.asarray(tauip, dtype=np.float64)
    powip = np.zeros(nlay, dtype=np.float64) if powip is None else np.asarray(powip, dtype=np.float64)

    ntout    = ctypes.c_int32(0)
    timesout = np.zeros(N_ARR, dtype=np.float64)
    respout  = np.zeros(N_ARR, dtype=np.float64)
    drespout = np.zeros((N_ARR, N_ARR), dtype=np.float64, order='F')

    tem1d_(
        ctypes.byref(ctypes.c_int32(imlm)),
        ctypes.byref(ctypes.c_int32(nlay)),
        _pad_to_n(rhon), _pad_to_n(depn),
        ctypes.byref(ctypes.c_int32(imodip)),
        _pad_to_n(chaip), _pad_to_n(tauip), _pad_to_n(powip),
        ctypes.byref(ctypes.c_double(txarea)),
        ctypes.byref(ctypes.c_double(rtxrx)),
        ctypes.byref(ctypes.c_int32(izeropos)),
        ctypes.byref(ctypes.c_int32(ishtx1)),
        ctypes.byref(ctypes.c_int32(ishtx2)),
        ctypes.byref(ctypes.c_int32(ishrx1)),
        ctypes.byref(ctypes.c_int32(ishrx2)),
        ctypes.byref(ctypes.c_double(htx1)),
        ctypes.byref(ctypes.c_double(htx2)),
        ctypes.byref(ctypes.c_double(hrx1)),
        ctypes.byref(ctypes.c_double(hrx2)),
        ctypes.byref(ctypes.c_int32(len(xpoly))),
        _pad_to_n(xpoly), _pad_to_n(ypoly),
        ctypes.byref(ctypes.c_double(x0rx)),
        ctypes.byref(ctypes.c_double(y0rx)),
        ctypes.byref(ctypes.c_int32(iresptype)),
        ctypes.byref(ctypes.c_int32(ideriv)),
        ctypes.byref(ctypes.c_int32(irep)),
        ctypes.byref(ctypes.c_int32(iwconv)),
        ctypes.byref(ctypes.c_int32(len(filtfreq))),
        ctypes.byref(ctypes.c_double(repfreq)),
        _pad_to_n(filtfreq, N_FILT),
        ctypes.byref(ctypes.c_int32(len(twave))),
        _pad_to_n(twave), _pad_to_n(awave),
        ctypes.byref(ntout),
        timesout, respout, drespout,
    )

    n = ntout.value
    nparm = nlay + 1 if imlm == 1 else 2*nlay
    return timesout[:n], respout[:n], drespout[:n, :nparm]
print("✓ tem1d_forward ready")
