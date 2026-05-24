# Schémas d'Ozaki pour l'émulation de précision

Projet de recherche - M1 CCA Sorbonne Université (2025-2026)

**Anatole SAINERO** & **Thomas VEY**

Under the supervision of **Théo MARY**

## Subject

The aim of this work is to analyze numerical properties and compare with current methods the Ozaki-I and Ozaki-II schemes.

These methods enable high-precision matrix multiplication for floating-point coefficients while preserving full information during computation. They also allow the use of lower-precision arithmetic, like 8-bit integer operations, without losing any accuracy.

The schemes implementations can be found in the files with the same names (`ozaki_1`, `ozaki_2`, `ozaki_2_hyb`). The `esc` file contains an implementation of the homonymous method, which is used to determine certain parameters of Ozaki 1.

You can also find some utilitary functions in files `utils`, `baseline` and `err_metrics`.

The jupyter notebook contains the code used to produce the graphs you can find in the report.

## How to run

To run this code, you need a python environment with the following modules :

- `numpy`, for matrix representation and fast computation
- `mpmath`, for multi-precision baseline
- `matplotlib`, for graphs
- `pytest`, for... testing
- `jupyterlab`, to run the jupyter notebook if needed
- `ml_dtypes`, for access to float8_e4m3 type

There is no requirements.txt as we think those modules are pretty common and the versions should not matter.

## Tests

To run tests, go to parent repository and type 'pytest' in terminal.

The tests are pretty basic and not even complete. They just allowed us to catch any issues when
modyfing the code.

## Additional info

> The implementation of the last version, Hybrid Ozaki 2 using FP8, does NOT work correctly.
