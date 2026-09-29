# Reviewer verification case: topology-aware matrix assembly

This repository is a self-contained numerical case for verifying the
topology-aware generalized mass/force assembly described in the manuscript
*Pywind: A Topology-Aware Software Framework for Efficient Nonlinear
Aeroelastic Simulation of Offshore Wind Turbines*.

It is intentionally independent of the full Pywind code base. The matrices,
partial velocities, forces, and branch topology are generated synthetically
from a fixed random seed. No turbine input files or proprietary production
modules are required.

## What this case verifies

The case represents three serial blade branches. For segment `i`, the ordered
active-coordinate set contains the shared upstream coordinates and the prefix
of coordinates belonging to that blade branch:

```text
k_i = u + i (6 + r),  i = 1, ..., s
```

where `u` is the upstream-coordinate count, `s` is the number of segments, and
`r` is the number of retained internal modes per segment. The local row count
is `p = 12 + r`.

Two algebraically equivalent paths are evaluated:

1. **Dense reference**: expand every local partial-velocity matrix to all
   global columns, then evaluate `V.T @ M @ V` and `V.T @ h`.
2. **Topology-aware assembly**: retain only the reachable columns, exploit the
   rigid/local row support, and scatter the symmetric local contributions by
   the stored global indices.

With `d = 6 + r` newly introduced coordinates and
`a_i = k_i - d` inherited coordinates, the local partial-velocity matrix has
the exact support pattern

```text
        [ V_i,R   0   ]
V_i,a = [             ],   V_i,R: 6 x a_i,   V_i,E: d x d.
        [   0    V_i,E]
```

The production blade coordinates make `V_i,E` an identity block. The
verification kernel keeps the more general block-supported `V_i,E` so that it
tests the support-aware multiplication without relying on unit entries.

Here, “sparse” refers to topology-induced structural zeros in the local
partial-velocity matrices. This is not a generic CSR/CSC reformulation. The
assembled generalized matrix remains dense, while unreachable columns and
zero row-column products are omitted before assembly.

The core operation is:

```text
for each branch and segment:
    A = ordered reachable global coordinates
    split A into inherited and newly introduced coordinates
    form W = M V using only each column group's supported local rows
    evaluate the upper triangle of V^T W
    scatter by A and mirror the off-diagonal entries
    scatter V^T h by A
```

The scripts check:

- generalized mass-matrix and force-vector equivalence;
- symmetry of the assembled mass matrix;
- the nested active-coordinate topology;
- storage scaling of dense and packed partial-velocity arrays;
- indicative assembly time for 5, 10, 25, and 50 segments;
- the effect of retaining three internal modes per segment.

This case verifies the assembly algorithm, not aeroelastic or hydrodynamic
physics. Timing depends on hardware and BLAS configuration; numerical
equivalence and array dimensions are the portable acceptance criteria.

## Quick start

Python 3.10--3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_case.py
python -m pytest -q
```

The default command runs the manuscript configurations:

```text
(segments, internal modes/segment)
(5, 0), (10, 0), (25, 0), (50, 0), (5, 3), (10, 3)
```

For a faster smoke test:

```bash
python run_case.py --quick
```

## Outputs

`python run_case.py` creates:

- `results/validation.json`: equivalence and topology checks;
- `results/benchmark.csv`: timing and deterministic array-storage metrics;
- `results/environment.json`: interpreter, dependency, platform, and run settings;
- `results/matrix_structure.png`: packed and zero-padded matrix structures;
- `results/scaling.png`: timing and storage scaling.

The validation command exits with a non-zero status if any numerical or
topological check fails.

The dense partial-velocity array is materialized before either path is timed.
Its construction cost is therefore excluded, making the reported comparison
conservative. Both methods assemble the same dense generalized matrix and
vector from the same local inputs.
Consequently, the reported timing ratio is the combined effect of reachable-
column packing, rigid/local row-support specialization, symmetric upper-
triangle evaluation, and direct scatter; it is not attributed to column
packing alone.

## Acceptance criteria

For every configuration:

```text
max(abs(G_packed - G_dense)) <= 1e-9
max(abs(f_packed - f_dense)) <= 1e-10
G_packed is symmetric within 1e-11
```

The tolerances allow for changes in floating-point summation order between the
Numba kernel and the BLAS-backed dense reference.

## Files

- `assembly_case.py`: deterministic case generator and both assembly paths;
- `run_case.py`: validation, benchmark, and plotting driver;
- `tests/test_assembly.py`: automated acceptance tests;
- `requirements.txt`: minimal runtime/test dependencies.
- `.github/workflows/verify.yml`: reproducibility checks for supported Python
  versions.

The committed files in `results/` are one reference run. Reviewers should
expect floating-point errors and array-storage values to agree within the
stated criteria; wall-clock timings will vary by machine.

## Disclosure boundary

This package exposes only the matrix-assembly method needed to reproduce the
algorithmic claim. Configuration handling, aero-hydro-servo-elastic coupling,
production data models, and the complete Pywind implementation are outside the
review package.
