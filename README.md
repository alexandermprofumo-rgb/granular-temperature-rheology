# Granular temperature and the inertial number in dense granular flow

Simulation and analysis code for a study of how the effective friction of a
dense granular flow depends on granular temperature at fixed inertial number,
in two and three dimensions.

The question is whether the two variables combine separably,

    mu_eff * Theta^n = F(I),

and if so what the exponent `n` is. From 2454 discrete-element runs: they do
not. Separability holds for frictionless grains and fails at every grain
friction tested, and the local slope `n` is roughly an order of magnitude below
the value a fabric-counting argument gives.

**Paper:** *Grain friction breaks the separability of granular temperature and
inertial number in dense granular flow.* A. Profumo. The manuscript source is
added here at submission; `paper/` currently holds the bibliography and the
figures.

---

## Layout

| Path | Contents |
|---|---|
| `src/` | Everything that runs: LAMMPS input decks (`in.*`), campaign drivers (`run_*.sh`), analysis and figure code (`*.py`), and the derived per-cell tables (`*.csv`). Flat on purpose, because the scripts resolve data files relative to the working directory. |
| `paper/` | Bibliography and figures. The LaTeX source is added at submission. |
| `ANALYSIS_PROTOCOL.md` | The pre-registered analysis plan and its seven dated amendments. |

Inside `src/`, the entry points are:

- `make_paper_figures.py` regenerates all six figures in the paper.
- `constraints_final.py` the constraint table.
- `nfit.py` gating, local-slope fits, the error model.
- `analyze_iscan3.py`, `analyze_iscan3d.py` the separability tests, 2D and 3D.
- `analyze_jamming_gate.py` the constraint list under three jamming thresholds.
- `check_pidamp.py` the thermostat-coupling null.
- `run_sweep.sh` and the other `run_*.sh` campaign drivers.

## Data, and what you need for what

Three tiers, because the raw output is far too large to distribute.

**1. Derived tables — in this repository, ~1 MB.**
The `*.csv` files in `src/`. These hold per-cell fitted values and are what
most of the analysis scripts consume.

**2. LAMMPS logs — Zenodo, 71 MB compressed.**
8994 per-run `log.*` files holding the time series, across 42 campaign
directories, plus LAMMPS' own `log.lammps` scratch file in each. **The figure
scripts need these**, not just the CSVs. Download the archive and unpack it
inside `src/`:

```bash
cd src
tar -xf /path/to/lammps-logs.tar.xz     # restores sweep_*/log.* alongside the code
python make_paper_figures.py
```

**3. Raw dumps — not distributed, ~198 GB.**
Per-run trajectory and per-contact dumps. Excluded by `.gitignore` and not
deposited. They are needed only to recompute the logs and the derived tables
from scratch, which means re-running the campaigns. Available from the author
on request for a specific campaign.

Three scripts read the per-contact dumps directly and therefore need tier 3:
`rb_tautology.py`, `analyze_tier1.py`, `analyze_strong_weak.py`. Without the
dumps they fail rather than skipping. Nothing in the paper's figure or
constraint pipeline depends on them.

## Reproducing the figures

Python 3.11+ with `numpy`, `scipy`, `matplotlib`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
tar -xf lammps-logs.tar.xz              # tier 2, from Zenodo
python make_paper_figures.py
```

Figures are written into `src/`. No LAMMPS installation required.

## Re-running the simulations

Requires a LAMMPS build with the `GRANULAR` package, with `lmp` on `PATH`.

```bash
PY=/path/to/python ./src/run_sweep.sh
```

Drivers take the interpreter from `$PY`, falling back to a project-local
`.venv` and then to `python3`. Campaigns are long: comparing at equal strain
makes step count scale as `1/gamma_dot`, so the slowest shear-rate arm costs ten
times the fastest, and the full set is 2454 runs.
The production set is listed run by run by `make_manifest.py`, which writes
`src/derived/manifest.csv` once the log archive is unpacked. The archive also
holds exploratory and superseded campaigns that no figure uses.

## The pre-registration

`ANALYSIS_PROTOCOL.md` was written 2026-08-19, before the queued campaigns
produced data. For each campaign it fixes the statistic, the decision
threshold, the predicted outcome, and the conditions that would count as fatal
to the main result. Seven dated amendments follow, each stating what changed
and why; two record results that were later withdrawn.

This is worth reading before the analysis code. Several choices in the code
look arbitrary until you see they were fixed in advance, and a few are
documented departures from the plan rather than the original design.

## AI use disclosure

Anthropic's Claude (Opus 5) was used in preparing the code in this repository.
It wrote and modified Python analysis and figure code, specifically the fitting
module, the four validity gates, and the scripts that produce the paper's
figures. It proposed the frozen-degree jackknife and the two-stage jamming
gate, which the author adopted after review and which are recorded as
Amendments 6 and 7 in `ANALYSIS_PROTOCOL.md`. It identified errors in earlier
drafts of the analysis, including two results that were subsequently withdrawn,
and it wrote several of the scripts that make previously undocumented claims
reproducible.

No LLM is listed as an author or co-author of this work.

The manuscript is not in this repository. It is added at submission, and it
carries its own statement of how Claude was used in preparing it, in the
acknowledgments. That statement covers the text; this one covers the code.

### Files substantially written or modified by Claude

| File | Scope |
|---|---|
| `src/nfit.py` | setpoint gating, adaptive-degree fitting, the frozen-degree jackknife and its analytic null, the degree systematic |
| `src/analyze_iscan3.py` | the three-arm separability test, the window-matching gate, the model-free sub-window slopes |
| `src/analyze_iscan3d.py` | the three-dimensional separability test and its per-arm constancy test |
| `src/make_paper_figures.py` | all six paper figures |
| `src/constraints_final.py` | the constraint table, adaptive and quadratic cross-check, split error reporting, the peak selection-bias calculation |
| `src/analyze_channel_state.py` | the thermodynamic against contact-stress slope comparison |
| `src/analyze_nothermo.py` | the unforced-locus block |
| `src/make_manifest.py` | the whole file |
| `src/run_seed3_2d.sh` | the third-seed campaign driver |
| `paper/lint.py` | the whole file |

Smaller edits, meaning interpreter selection, script-relative paths, output
directories, missing-data guards and docstring corrections: `src/jk_calib.py`,
`src/jk_calib2.py`, `src/jackknife_null_test.py`, `src/rb_tautology.py`,
`src/analyze_tier1.py`, `src/analyze_strong_weak.py`, `src/analyze_ktgrid.py`,
`src/build_master_table.py`, `src/analyze_jamming_gate.py`,
`src/check_3dconv.py`, `src/check_iscan.py`, `src/check_pidamp.py`,
`src/run_bigN.sh`, `src/run_finite_size.sh`, `src/run_levers_steady.sh`,
`src/run_stiff3d.sh`.

Files not listed above predate this work or were written by the author.

## Citing

See `CITATION.cff`.

## License

MIT, for the code and the derived tables. See `LICENSE`.
