# Charge states of a near-critical Coulomb impurity in gapped graphene

Code, data and figures for

> E. Hocine, *Charge states of a near-critical Coulomb impurity in gapped graphene: Correlation
> and gate control* (manuscript, 2026; REVTeX/PRB format)

Companion to *Atomic collapse in gapped two-dimensional Dirac materials: Screening by the Dirac sea,
gates, and resonance widths*, whose single-particle code `collapse2d`
([ehocine/dirac-collapse](https://github.com/ehocine/dirac-collapse), release 1.0.0,
[doi:10.5281/zenodo.23123320](https://doi.org/10.5281/zenodo.23123320)) this package builds on.
Repository: https://github.com/ehocine/impurity-charge-states · Archive (all releases; 1.0.5 is the version of the manuscript): [doi:10.5281/zenodo.23249678](https://doi.org/10.5281/zenodo.23249678) · License: MIT (see `LICENSE`) · Citation: `CITATION.cff`

## Quick start

```bash
pip install -r requirements.txt                      # numpy, scipy, mpmath, matplotlib, pytest
git clone --branch 1.0.0 https://github.com/ehocine/dirac-collapse external/dirac-collapse
make test                                            # 29 regression tests (~7 min)
make figures                                         # all figures and generated tables from results/ (~1 min)
```

`collapse2d` is found automatically in `external/dirac-collapse`; otherwise set `COLLAPSE2D_ROOT`
(see `chargestates/deps.py`). The production runs that produce `results/` take several hours on a
laptop (see below); the figure scripts only read the saved results.

## Model and method in brief

No-pair (Furry-picture) Hamiltonian for up to four electrons bound to a charge Ze at 0.3 nm above
gapped graphene, with the Dirac-sea (RPA) screened impurity potential and electron-electron
interaction of the companion paper, optional metallic gate. Basis: exact box eigenstates of the
radial Dirac equation, found by Pruefer-angle counting; above Z_c the dived level is kept as a
quasi-bound orbital while it is a narrow resonance. Two-, three- and four-electron configuration
interaction with a matrix-free Davidson solver and q-compressed interaction form factors.
Low magnetic fields (Table IV): field in the radial Dirac equation and in the Furry projector.

## Layout

```
chargestates/
  deps.py          locates collapse2d (release 1.0.0)
  furry.py         box Furry basis: Pruefer counting, eigenfunctions, quasi-bound dived levels
  interaction.py   screened interaction W(q), multipole form factors, matrix elements, q compression
  ci.py            two-electron CI (same-valley singlet/triplet, K-K' pair), Davidson, Dyson orbitals
  ci3.py           three-electron CI (K pair + K' electron), Davidson, Dyson orbitals; optional
                   separate basis for the K' electron (valleys differ in a field)
  ci4.py           general CI for two to four electrons with any valley content (four-electron
                   sectors, rigorous lower-bound sector skipping, memory-lean Davidson)
  field.py         Furry basis in a magnetic field (Pruefer-index projector, K' box G(R) = 0)
  meanfield.py     Hartree potential (fixed q nodes + Ogata quadrature), mean-field basis
  observables.py   orbital magnetic moments, polarization self-energy estimates (single mode and
                   RPA electron-hole continuum of the gapped sea), moire form factors,
                   intervalley exchange estimate
tests/             test_m1.py (basis, interaction), test_m2.py (two-electron CI),
                   test_m3.py (three-electron CI, Hartree, dived level), test_observables.py,
                   test_field.py (field basis, Landau levels, low-field moment, two-basis CI),
                   test_ci4.py (general CI against ci/ci3, four-electron checks)
scripts/
  m2_ci_convergence.py, m2_emax_series.py   Table II   convergence (cut-off, multipoles, box); D- check
  m3_two_electron_scan.py                   one- and two-electron energies vs Z (results/m3/scan_*)
  m3_three_electron_scan.py                 three-electron energies vs Z (results/m3/three_*)
  m4_diagram.py                             Fig. 1     charge-state diagrams
  m4_first_order.py                         Fig. 2     first-order addition energies
  m4_zc1.py                                            Z_c^(1) (mean field + correlation correction)
  m4_sts.py                                 Fig. 4, Table V   Dyson orbitals, quasiparticle weights
  m5_figures.py                             Figs. 2, 3 addition energies, valley splitting
  m6_estimates.py                           Tables III, IV, App. C   moments and field slopes (Table IV
                                            from results/m6/finite_field_moments.json),
                                            self-energy, moire, intervalley exchange
  m7_finite_field_moments.py                Table IV   finite-field moments -dE_N/dB at 2, 4 mT
  m8_four_electron_scan.py                  Fig. 1     four-electron energies vs Z (results/m8; --L sets
                                                       the channel window)
  m9_sector_gaps.py                         Sec. V     lowest energy of every three-electron sector
                                                       vs Z (results/m9)
  m10_three_complete.py                     Sec. III   three-electron addition energy in the complete
                                                       basis at several E_max (results/m10)
  m11_gate_kernel.py                        Sec. II    sensitivity to the gate model: homogeneous kappa vs
                                                       the exact vacuum/hBN/metal kernel (results/m11)
  u_estimate.py                             first-order charging energy (planning estimate)
results/           JSON outputs; results/m3/run_*.sh reproduce the scans (order: run_scans,
                   run_three, run_batch2, run_gaps, run_nmax_all, run_nmax_check)
figures/           figures of the manuscript
paper/tables/      tables generated by scripts/m6_estimates.py
```

## Reproducing the data

| step | command | time (laptop) |
|---|---|---|
| convergence, Table II | `scripts/m2_ci_convergence.py`, `results/m2_series/run*.sh` | ~2 h |
| scans vs Z | `results/m3/run_scans.sh`, `run_three.sh`, `run_batch2.sh`, `run_gaps.sh` | ~4 h |
| mu3 vs n_max (extrapolation in Fig. 1) | `results/m3/run_nmax_all.sh` (n_max = 12, 16), `run_nmax_check.sh` (24, 28) | ~2 h |
| analysis | `python3 scripts/m4_first_order.py; python3 scripts/m4_zc1.py; python3 scripts/m4_sts.py` | ~1.5 h |
| four electrons, Fig. 1 | `results/m8/run_conv.sh`, `run_conv_all.sh` (n_max = 12, 16, 20; run one large case at a time, ~1-2 GB each at n_max = 16) | ~10 h |
| finite-field moments, Table IV | `python3 scripts/m7_finite_field_moments.py` | ~15 min |
| estimates, Tables III-IV | `python3 scripts/m6_estimates.py` | ~20 min |
| figures | `make figures` | ~1 min |

Set `CI_DTYPE=float32` for the large runs (form factors in single precision; energies change by
< 1e-8 Delta).
