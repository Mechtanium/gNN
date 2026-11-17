# Parity with PINN-Lab

The numerics under `modules/` are the research package `pinnlab` with every branch the three work paths never take removed and nothing else changed. Two developer scripts prove it: `upstream_path.py` trains `pinnlab.config.autopilot` with the additions a work path makes, `extracted_path.py` trains the same work path through `modules.autopilot`, and the second compares every loss of every iteration and the parameters' checksum with the first's JSON.

```bash
# the research path, with the research repository's interpreter (writes the target)
<research>/.venv/bin/python tests/parity/upstream_path.py <research root> SPE1CASE1 history 5 up.json
# the same work path here, on copies of the research repository's prep and spectral caches
mkdir -p work && cp -a <research>/output/SPE1CASE1/{prep_cache,spectral_cache} work/
.venv/bin/python tests/parity/extracted_path.py <research>/data/spe1/case1/SPE1CASE1.DATA work history 5 up.json
```

`history` is work path 1 (cell states and well rows at weight 0.1); `forecast:<n>` is work path 2 with `n` report steps of history (predicted wells with the three-state switch, phase-equilibrium and material-balance rows). Both read the research repository's extraction and eigenbasis, because the eigenvectors of a degenerate spectrum are defined only up to a rotation, so two solves of the same problem differ; with them pinned the comparison is exact. They are developer checks, not part of the pytest suite.

Last result (2026-11-17, CPU, five iterations, SPE1CASE1): **IDENTICAL** for both.

- `history`: totals `[2.2251, 0.49175, 0.45252, 0.25720, 0.12317]`, groups `pde+ic+data+well`.
- `forecast:61`: totals `[6.0909, 5.1026, 2.8991, 2.6470, 1.2210]`, groups `pde+ic+data+well+ctrl+eq+mb`.
