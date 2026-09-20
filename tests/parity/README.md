# Bit-identity check against PINN-Lab

`notebook_path.py` runs the research repository's `pinnlab` pipeline and `extracted_path.py` runs this repository's `modules` pipeline on the same deck, configuration and seed, and the second compares its loss trajectory and parameter checksum with the first's JSON. Both need the research checkout (`/home/enosmath/Flintstone/Delta-PINNs`), its venv and its SPE2EQUI prep cache; the work directory for the extracted path holds symlinks to that cache (see the scripts). They are developer checks, not part of the pytest suite.

The eigenbasis must be pinned for the comparison to mean anything: `resolve_n_eig` re-solves the candidate pool on every run and the eigenvectors of this symmetric grid are only defined up to a rotation inside each degenerate pair, so two runs of the *same* code differ unless they load the same cached basis. Both scripts therefore set `n_eig=3` with `spec.addressability_min=0.0`, which loads `eigenbasis.npz` and skips the re-solve.

Last result (2026-09-20, CPU, 5 iterations, seed 30): IDENTICAL — totals `[159.2455, 4.1925, 3.8617, 3.7266, 2.1870]`.
