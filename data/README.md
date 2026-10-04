# Field data (not distributed)

The field data used in the manuscript are not included in this repository. To run steps 03 and 04 with your
own data, place files with the following names and formats in this folder (or point `TEM_DATA_DIR` at the folder):

| File | Used by | Description |
|------|---------|-------------|
| `June data.usf` | steps 03, 04 | Soundings with LM + HM gates, quality flags and error bars (Aarhus .usf format) |
| `Line001.xyz`   | steps 03, 04 | Aarhus SPIA smooth-inversion export (one row per layer per station); used as the comparison model |
| `LM40.csv`, `HM40.csv` | step 00 only (optional) | Two-column CSV (time, dB/dt) for the forward-model demo |
