# TEM1D Fortran library (not included)

The 1D forward modelling uses the open-source **TEM1D** FORTRAN subroutine, which is not redistributed here:

> Christensen, N. B., Christiansen, A. V., Auken, E., Foged, N. (2026). An open source FORTRAN subroutine for
> calculation of TEM responses and derivatives from 1D models. *Computers & Geosciences*, 209, 106102.
> https://doi.org/10.1016/j.cageo.2025.106102

Source code and manual (MIT licence): https://github.com/hydrogeophysicsgroup/TEM1D

Please cite the paper if you use this code or its results.

## Build (Linux, gfortran)

```bash
git clone https://github.com/hydrogeophysicsgroup/TEM1D
cd TEM1D
gfortran -O2 -fPIC -shared -std=legacy -w -o libtem1d.so \
    TEM1D.for TEM1DFHT.for TEM1DFUNC.for TEM1DRESP.for \
    TEM1DRESPIP.for TEM1DRESPPOLY.for TEM1DRESPPOLYIP.for
cp libtem1d.so /path/to/this/repo/fortran/      # or: export LIBTEM1D=/path/to/libtem1d.so
```

On Windows build a `.dll`, on macOS a `.dylib`, and point the environment variable `LIBTEM1D` at the result.
`TEMTEST.for` in the authors' repository is their test driver and is not part of the library.
Tested with GNU Fortran 13.3 against the repository state of January 2026 (commit `7b91ad9`); the library exports
the symbol `tem1d_` that `tem1d_cnf/tem1d_wrapper.py` calls through `ctypes`.

## Consistency with the wrapper

The wrapper assumes `N_ARR = 512` (`ARRAYSDIMBL.INC`) and `N_FILT = 16` (`FILTFREQ(16)` in `TEM1D.for`),
64-bit reals and 32-bit integers. If you change `N_ARR` in the Fortran sources, change it in the wrapper too.
