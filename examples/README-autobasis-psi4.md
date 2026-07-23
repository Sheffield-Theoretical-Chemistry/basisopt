# Running the auto-basis pipeline start-to-finish with Psi4

This walks through generating a contracted atomic basis for oxygen entirely on
the **Psi4** backend — including the **natural-orbital (NAO) contraction**, which
Psi4 now generates natively (no Molpro needed).

Files:
- [`autobasis-O-psi4.yaml`](autobasis-O-psi4.yaml) — the run config (all six steps, Psi4)
- [`data/O2.xyz`](data/O2.xyz) — the diatomic geometry the uncontraction step needs
- [`nao_from_scratch.py`](nao_from_scratch.py) — the annotated, standalone NAO walkthrough

## Prerequisites

- Psi4 installed in the environment (`python -c "import psi4"`).
- Edit the two `>>>` values in the config for your system:
  - `reference.cbs_limit` — your extrapolated atomic CBS limit (Eh),
  - `reference.geometry` — path to the diatomic xyz (an O₂ is provided).

## Run it

From the repo root:

```bash
# run every step listed in the config
python -m basisopt.autobasis run examples/autobasis-O-psi4.yaml

# see what has completed
python -m basisopt.autobasis status examples/autobasis-O-psi4.yaml
```

Everything lands under the config's `workdir` (`./runs/O-psi4`). Each step writes
its own numbered folder:

```
runs/O-psi4/
  manifest.json                 # what has run (used for resume)
  config.resolved.yaml          # the merged config, for provenance
  01_primitives/   basis.json  record.json  basis.molpro.txt
  02_reduction/    ...
  03_contraction/  ...          # <- the NAO contraction
  04_uncontraction/ ...
  05_purification/ ...
  06_pruning/      ...
```

`basis.json` is the internal basis after that step; `record.json` holds the
diagnostics.

## What the six steps do

1. **primitives** — lay down primitive exponents from a Legendre expansion.
2. **reduction** — drop the least-important exponents down to `reference.target`.
3. **contraction** — **natural-orbital contraction** (see below).
4. **uncontraction** — re-free functions until the decontraction error is within
   `decontract_error_percent`.
5. **purification** — clean up the contraction coefficients (extended Davidson).
6. **pruning** — zero contraction coefficients within an energy budget.

## The NAO contraction (step 3), per backend

`contraction.generate: true` tells the step to **generate** the NAOs with the
active backend rather than adopt an externally-produced file. The interface is
per-backend — each program builds NAOs its own way, behind a common method:

```python
wrapper.natural_orbitals(atom, params)  ->  {l: (occupations, coefficients)}
```

- **Psi4** (`Psi4Wrapper.natural_orbitals`) uses the *density-average route*: run
  (U)KS on the uncontracted atom, form `D = Da + Db` and the overlap `S`, average
  each shell over its `2l+1` m-components (this restores the spherical symmetry an
  open-shell atom breaks), then take the natural orbitals in the `S` metric
  (Löwdin). The leading orbitals per shell — how many is set by
  `contraction.n_keep`, e.g. `{s: 2, p: 1}` for `[2s1p]` — become the contraction
  coefficients. This reproduces the Molpro averaged NAOs (validated: O `8s6p→[2s1p]`
  overlaps Molpro's to 0.999+; closed-shell Ne is exactly lossless).
- **Molpro** (native averaged-NAO output) can be added later by implementing the
  same `natural_orbitals` method on `MolproWrapper`; nothing else in the pipeline
  changes.

`03_contraction/record.json` reports the kept `occupations` per shell and, when
`evaluate_energy: true`, the `contraction_error_mEh` (contracted vs uncontracted
energy) — the direct quality check. The standalone
[`nao_from_scratch.py`](nao_from_scratch.py) prints the same numbers step by step
if you want to see the mechanism in isolation.

## Iterating

The run is resumable via `workdir/manifest.json`, so you can redo one step
against the same workdir without repeating the others — just set `steps:` to the
subset you want:

```yaml
# e.g. trial a different contraction size without redoing steps 1-2
steps: [contraction]
contraction: { generate: true, n_keep: {s: 3, p: 2} }
```

```bash
python -m basisopt.autobasis run examples/autobasis-O-psi4.yaml
```
