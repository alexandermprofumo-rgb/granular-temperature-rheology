# Frozen analysis protocol

**Written 2026-08-19, before queued campaigns produced any data,** with
`sweep_size2d`, `sweep_restit` and `sweep_iscan2` empty.

## Why this exists

Four methodological corrections in sequence: strain 0.12 → 1.0, the jamming
gate, the calibrated jackknife, and measured a_t. Every one
was triggered by a demonstrated defect rather than by a preference for a
different answer, and the corrections shrank (strain moved n by up
to 0.06; the jamming gate by ~0.01; the error calibration moved errors, not
values).

Nothing below was changed after looking at the results. If something here
turns out to be wrong, the fix is a labelled amendment with a date and
reason.

---

## 1. Simulation policy

| | |
|---|---|
| strain | 0.5 equilibration + 0.5 measurement, **strain not step count** (steps = strain/(γ̇·dt) per cell) |
| seeds | 2, independent packings |
| measurement | last 2/3 of the measurement window (`nfit.parse_log`) |
| dumps | tangential force always (`p1 p2 p4`); a_t and χ measured, never closed |

## 2. Validity gates — all four, applied per setpoint, in this order

1. **Barostat** σ_P/⟨P⟩ ≤ 0.15
2. **Fixed-I** |⟨I⟩/median − 1| ≤ 0.03
3. **Thermostat** ⟨Θ⟩ ≤ setpoint, and ⟨Θ⟩ strictly increasing in it
4. **Jamming** mean Z ≥ D+1 (3 in 2D, 4 in 3D)

A cell needs ≥ 4 surviving setpoints to be quoted at all.

Gate 4 is new and is the one most likely to be argued with. It is set at the
*frictional* isostatic bound, not the frictionless one (2D and 3D respectively),
because frictional packings legitimately jam below the frictionless value —
gating at 2D kills every frictional cell. Sensitivity to this choice will be
reported alongside any result that depends on it.

## 3. Estimator

`ln µ = a + b·ln(Θ/Θ₀) + c·ln²(Θ/Θ₀)`, and **n(Θ₀) = −b**.

- **Θ₀** is the geometric centre of the Θ range common to *every cell in the
  comparison being made*, so no value is an extrapolation. It is recomputed per
  comparison and always quoted.
- **stat** = OLS error on b.
- **sys** = the setpoint jackknife's **excess over its own null expectation**,
  `sqrt(max(jk² − null², 0))`, with the null computed analytically
  (`nfit._jk_null_factor`) rather than assumed. The raw jackknife is not a
  systematic; it has a null expectation of ~1.3 × stat on these designs.
- **tot** = `hypot(stat, sys)`. Every quoted number carries tot.
- Cells where no systematic can be formed report `sys_ok = False` and are **not
  quotable**.

## 4. Slope fits

Weighted line, covariance `inv(AᵀWA)` propagated from the per-cell errors,
**never rescaled by residuals**. χ²/dof reported alongside; where χ²/dof > 1 the
error on any derived quantity (e.g. a zero-crossing) is inflated by √(χ²/dof).
Linear fits are applied only over ranges where the quantity is monotonic, and
the range is stated.

## 5. Pre-stated tests for the queued campaigns

Each names its statistic and its threshold **before** the data exists.

### 5.1 Finite-size scan (`sweep_size2d`, N = 1000/2000/8000 vs 4000)

- **Statistic**: weighted slope of n against ln N, at a common Θ₀, per friction.
- **Pass** (no finite-size effect): |slope| < 2σ at all three frictions.
- **Fail**: any friction ≥ 2σ. If it fails, N = 4000 is not converged and every
  constraint inherits a size systematic that must be quoted.
- **Prediction on record**: pass. n is a local log-slope of a stress ratio, and
  stress ratios converge fast in N.

### 5.2 Restitution scan (`sweep_restit`, e = 0.1/0.5/0.9)

- **Statistic**: weighted slope of n against e, per friction.
- **Pass**: |slope| < 2σ.
- **Prediction on record**: no strong opinion. A real dependence would be a
  reportable finding rather than a problem.

### 5.3 I-scan and F(I) (`sweep_iscan2` + the existing γ̇ = 1e-3 arm)

- **∂n/∂ln I is reported per friction.** A global weighted mean is quoted
  **only if** χ²/dof < 2 across frictions. The previous campaign had
  χ²/dof = 3.6 and quoted the mean as "5.7σ" anyway; that will not be repeated.
- **F(I) collapse**: test whether µ·Θⁿ depends only on I, by comparing a shared
  exponent against free per-arm quadratics (nested F-test). Report the RMS
  penalty of the ansatz per friction. No threshold is set because this is
  descriptive, not a hypothesis test.

### 5.4 Rotational temperature Θ_rot (free, from the above runs)

Θ_rot is a **candidate state variable** and gets exactly the same test as the
nine already falsified — no special treatment because it is the newest idea and
the only one that could give a positive result.

- **Statistic**: the same two-lever/collapse machinery. Does n collapse against
  Θ_rot across levers that move Θ_rot by an orthogonal route?
- **Pass**: χ²/dof ~ 1 for a single n(Θ_rot) curve **and** no residual trend
  with the lever variable at ≥ 2σ. Both conditions, as for χ.
- **Prediction on record**: fail. The frozen-rotation test in the original
  report found the rise in n survives when rotations are frozen, which is
  evidence against rotation carrying the mechanism — though that was
  strain-0.12 data and is exactly why it is being re-tested.

## 6. Standing rules

- **No statistic may be selected after seeing the residuals.** §7a went
  RMS → coherent offset → paired jackknife (1.6σ → 2.1σ → 3.9σ), choosing the
  statistic each time after seeing the previous one fail. Where a pre-stated
  statistic returns null and a different one would not, both are reported and
  the second is labelled post-hoc.
- **Multiplicity is stated.** Ten candidate state variables have now been
  tested; any single one clearing 2σ must be read against that.
- **Nulls are quoted with their power** — the bound on the effect size
  excluded, not just the σ.
- **Every claim states its dimension.** Constraints 7–10 and all lever results
  are **2D only**; the channel decomposition and constraints 1–6 cover both.

## 7. What would overturn the current headline

Recorded so it cannot be quietly explained away:

| finding | what would kill it |
|---|---|
| n(µg=0) ≈ 0.02, 1/(2D) excluded | a finite-size trend that carries n(µg=0) up by ~0.1; or the exclusion falling below 3σ at any Θ₀ inside the jammed window |
| dimension-independence below µg ≈ 0.2 | 2D/3D differing by > 2σ at any friction ≤ 0.2 after finite-size correction |
| no state function for n | Θ_rot passing both conditions in §5.4 |
| the frictional-origin reframe | n(µg=0) proving to be an artefact of the jamming gate — i.e. moving > 2σ under a gate at D+1 vs 2D |

---

# Amendments

## Amendment 1 — 2026-08-25. Compliance corrections (not policy changes)

The second audit found three places where the shipped code did not implement
what §2 and §4 already required. These are fixes, not amendments, and are
recorded here only so the change in the numbers is traceable.

**1a. The four gates were not applied in the channel analyses.**
`analyze_channel_state.py` and `analyze_lev_E3d.py` applied none of the
barostat / fixed-I / thermostat gates, and applied the jamming gate on the
cell-**mean** Z rather than per setpoint. §2 requires all four, per setpoint,
in order. Fixed. The effect is large — ungated, C_c at µg = 0.12 fits to −0.25
against +0.063 gated — but no verdict changes: the 2D test goes from 0 passes
to 0 passes.

**1b. Trend statistics were not inflated by √(χ²/dof).** §4 requires it wherever
χ²/dof > 1. Both channel scripts now do. Consequence: **no residual trend
anywhere in the channel tables is significant, in either dimension.** The
failures are χ² failures only, and should be described that way. The 3D table's
"n_c 3.4σ, n_n 6.8σ" becomes 1.6σ and 1.7σ.

**1c. Hand-transcribed numbers.** Earlier working notes and the docstring of
`analyze_lev_E3d.py` carried 2D values (χ²/dof 1.1 and 0.8 "passing", 39.4 and
15.7σ for constraint 8) that reached them by transcription and were never
recomputed after the 2026-08-20 re-extraction. They are not reproducible under
any frame policy, fabric weighting, gate or Θ₀ — the best value obtainable over
84 configurations is 2.12 for n_c, still failing. `analyze_channel_state.py`
now writes `channel_state_2d.json` and `analyze_lev_E3d.py` reads it.
**Standing rule added: no number may reach a table or a docstring except from a
file a script wrote.**

## Amendment 2 — 2026-08-25. Θ WINDOW, not only Θ₀

**What changes.** §3 fixes Θ₀ for a comparison but is silent on the Θ *range*
fitted. It should not have been. With a curved µ(Θ), established at up to
12.7σ per cell, the local slope returned by a quadratic
at a fixed Θ₀ still depends on the window it was fitted over. Two cells
compared at a common Θ₀ but fitted over different ranges are not comparable.

**The rule, from this date.** Any comparison between cells must either (a) fit
both cells over the Θ range they share, or (b) quote the window systematic,
measured as the shift in n when the cell is refitted over the shared range at
the same Θ₀.

**Why it is an amendment and not a correction.** The principle was applied to
the I arms in `analyze_iscan3.py` before this date and recorded as a caveat
for future work. It was never applied to the
constraint list. Applying it now is a change of policy after seeing results, so
it is recorded as such and **both columns are reported** — never the matched
one alone.

**What it does to the numbers**, from `constraints_final.py`'s new window block:

* The 2D column is unaffected. 2D windows are the narrower ones in every 2D/3D
  pair, so 2D cells lose no setpoints.
* The **3D** column carries an unquoted window systematic of **1.0–3.6× its own
  published total error** in every frictional cell (worst: µg = 0.15, n moves
  0.1980 → 0.2485, 3.6× the quoted error).
* Constraint 2 (n ≠ 1/(2D) by magnitude) is unaffected in 2D (22σ) and remains
  excluded in 3D under every variant.
* Constraint 2b (the dimension *difference*) is not robust: 5.1σ as published,
  2.6σ matched, 1.4σ matched with the adaptive degree. It must be demoted from
  a headline to a sensitivity.
* **The dimension effect inverts.** With conservative errors in both columns:
  µg ≤ 0.2 gives χ²/dof 1.28 own-window against **7.87** matched; µg > 0.2
  gives 11.46 own against **2.52** matched. "Dimensions agree below µg ≈ 0.2
  and diverge above" is an artefact of the unmatched window and must be
  withdrawn.

**Errors in the matched column are the raw, uncalibrated jackknife**, because
restricting the window costs setpoints, which raises the null expectation and
truncates the calibrated excess to zero — at µg = 0.15 and 0.2 in 3D that makes
the matched error *smaller* than the unmatched one and flatters the comparison.
The raw jackknife has no such truncation, so both columns sit on one footing.

## Amendment 3 — 2026-08-25. §5.3's nested F-test is the primary separability statistic

§5.3 specified "a shared exponent against free per-arm quadratics (nested
F-test)" and "the RMS penalty of the ansatz per friction". That test was never
run. What was reported instead — dn/dlnI evaluated at 25%, 50% and 75% of the
window — is Θ₀-dependent by construction, and the two entries quoted in the
executive summary were the significant ones (11.1σ at the minimum-variance
point; 4.5σ at 25% of the window, where the pre-registered Θ₀ of §3 gives
1.7σ). The pre-registered test is now implemented as T0 in
`analyze_iscan3.py` and is the number to quote:

| µg | nested F | p | RMS penalty | β = γ = 0 | p | on setpoint means | p |
|---|---|---|---|---|---|---|---|
| 0 | 0.29 | 0.88 | 1.01× | 0.20 | 0.82 | 0.28 | 0.76 |
| 0.15 | 16.31 | 4.2×10⁻⁷ | 1.80× | 16.13 | 1.4×10⁻⁵ | 9.70 | 2.6×10⁻³ |
| 0.3 | 32.18 | 2.8×10⁻¹⁰ | 2.33× | 62.17 | 9.5×10⁻¹² | 51.90 | 6.3×10⁻⁷ |

It is stronger than what it replaces, it is immune to the Θ₀ objection, and it
recovers µg = 0.15 as a robust failure rather than a low-Θ one. µg = 0 remains
separable, which is the internal null control.

## Amendment 4 — 2026-08-25. §7 is withdrawn

The channel-level state-function result ("n_c and n_n collapse onto χ while n
does not") does not reproduce and is not recoverable — see 1c. Across 2D and 3D
the test is now **0 passes in 64 (channel, candidate) pairs**, fewer than the
~3 expected by chance at 2σ. The correct statement is that no microstructural
variable organises n *or any of its channels*, in either dimension. §6.1 and §7
of the earlier working notes collapse into one claim, and the tension between
them disappears.

## Amendment 5 — 2026-08-25. Two campaigns added post-hoc, with predictions on record

Both were queued on 2026-08-25 in response to audit gaps, before any of their
data existed. Predictions are recorded here for the same reason §5 exists.

**5a. `sweep_nothermo2` — the athermal control.** Removes the Langevin bath
entirely; Θ settles where shear heating balances inelastic dissipation, and
restitution scans Θ at fixed I. Overlaid on `sweep_restit` at matched e, µg,
γ̇ and packing. *Statistic*: do the athermal (Θ, µ) points lie on the
thermostatted µ(Θ) curve? *Pass*: within 2σ of the thermostatted fit
extrapolated to Θ_shear. *Prediction on record*: pass — but note the deck's own
caveat that agreement is informative and disagreement is ambiguous between "the
thermostat distorts" and "e affects µ directly".

**5b. `sweep_iscan3d` — the 3D I-scan.** The separability failure is currently
2D-only; every 3D run in the project sits at one shear rate. Two outer arms a
decade apart in I, sharing `sweep_steady3d`'s packings. *Statistic*: the same
T0 nested F-test as Amendment 3, per friction. *Prediction on record*:
separability fails at µg = 0.3 and holds at µg = 0, as in 2D. If 3D instead
holds at every friction, the form failure is 2D-specific and the paper's title
claim must say so.

## Amendment 6 — 2026-08-26. Adaptive degree is the primary fit

Amendment 2 left open which of its two remedies leads. Decided: **the
adaptive-degree fit is primary and the matched window is the cross-check**;
both continue to be reported, per Amendment 2.

*Rationale, in the order it carries weight.* Adaptive degree on own windows
already withdraws the dimension effect (χ²/dof 7.61 below µg = 0.2, 4.80 above)
without any window matching, so the conclusion does not rest on the more
contestable operation. Only the adaptive fit can produce a single table at one
Θ₀ — the window common to all 16 cells is 3.2×, too narrow for a quadratic. And
the selection rule (`nfit.fit_local_adaptive`: sequential F-test, α = 0.05, max
degree 4) is fixed in code from before the second audit, so it is not selection
on these results.

*Scope.* Moves constraint 2b, the dimension effect, and the rise/fall
significances of constraints 3 and 4. Does not move constraint 2 (22.3σ under
all four treatments) or the §5.3 separability result, which is measured within
2D on arms sharing one deck and one pair of packings.

*Recorded as a policy choice made after seeing results*, as Amendment 2 was.
The pre-audit constraint values remain in the record as the "quadratic, own
windows" column.

## Amendment 7 — 2026-08-28. The jamming gate is per-RUN, with a majority rule

**What changes.** Section 2 applies all gates per SETPOINT. For the jamming
gate that was a coarsening, and it let contaminated runs through. From this
date the jamming gate is applied in two stages:

1. **per run** — an individual run with window-averaged Z below the isostatic
   count is discarded;
2. **per setpoint** — the setpoint survives only if a STRICT MAJORITY of its
   runs did, and it then carries just those runs.

**Why this gate may be per-run when the others may not.** The module header of
`nfit.py` forbids per-run gating, and correctly: the fixed-I and thermostat
gates are RELATIVE, comparing a setpoint against a global median, so applying
them per run lets members of a deviant setpoint survive and can flip a fitted n
negative. The jamming gate is not relative. Z >= D+1 is an ABSOLUTE physical
statement -- below it the packing is not rigid and its "friction" is not the
quantity being fitted. That is true of a single run whatever its partners did.

**Why the majority rule is not optional.** Per-run filtering alone
cherry-picks. At a setpoint straddling the threshold it keeps whichever
realisation landed a hair above and admits a state the mean gate correctly
rejected. Surveyed over all ten sweeps: run-filtering alone removes 5 genuinely
bad runs and wrongly admits 10 such setpoints -- twice as much harm as good.
Every one of the 10 is a 1-of-2 or 1-of-3 survivor within +/-0.02 of threshold.
A minority surviving is the unjamming signal the setpoint gate exists to catch,
not a bad realisation to prune.

**What triggered it.** The third 2D seed (`run_seed3_2d.sh`, 81 runs) exposed
4 runs in `sweep_steady2d` and 1 in `sweep_restit`, out of 531, that failed to
jam at the coldest setpoint -- Z ~ 2.6 against partners at ~4.1, carrying mu
19-27% high. The setpoint mean averaged 2.6 with two 4.1s, cleared the
threshold, and admitted them. They sit at the coldest Theta, the
highest-leverage end of the fitted range.

The contamination was present with two seeds and invisible: six points support
only a quadratic, and a parabola is too stiff to chase one outlier, so it
averaged over it and returned a plausible number. Nine points justified degree
4, and a quartic is flexible enough to chase it -- mu_g = 0.1 went to
n = 0.041 +/- 0.175 and mu_g = 1 flipped sign. The third seed did not create
this; it revealed contamination that had been biasing the two-seed numbers
invisibly, which is the argument for running it.

**THIS CONDITIONS ON AN OUTCOME** and must be reported as such in the paper: we
keep runs that jammed and discard runs that did not. The justification is that
an unjammed run is not a measurement of the target quantity at all -- the same
argument that justifies the gate existing in the first place. The rate is to
be quoted: **5 runs of 531**.

**Scope.** Moves `sweep_steady2d` (the constraint list) and one `sweep_restit`
cell. `sweep_iscan2` and `sweep_iscan3d` contain ZERO affected runs, so the
section 5.3 separability result -- the title claim -- sits on identical data
before and after. Applied uniformly across all nine analysis entry points, not
because the data changes outside `steady2d` but because two analyses gating the
same runs differently is the inconsistency the second audit caught twice.
