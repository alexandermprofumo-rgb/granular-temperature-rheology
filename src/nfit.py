"""
nfit -- the single source of truth for fitting the exponent n.

WHY THIS EXISTS. n has been computed in this project by at least six code
paths (ncurve, gate_and_fit, analyze_anomaly, analyze_master, and inline
variants) with different gate thresholds and different temperature windows.
They disagree. Worse, the disagreement is not visible in the quoted errors:
at Pconf=10, mu_g=0.1, fitting 4 setpoints gives n = 0.1669 +/- 0.0096 and
fitting 7 gives 0.1865 +/- 0.0116 -- the same simulations, the same gates, a
shift of 2.0x the statistical error. mu(Theta) has real curvature, so the
fitted slope depends on which part of the curve you sample.

Every published number must therefore come from ONE policy, and must carry a
SYSTEMATIC error alongside the statistical one.

THE POLICY
Gates (all per-SETPOINT, never per-run -- gating individual runs against a
global median lets members of a deviant setpoint survive and can flip a
fitted n negative):

  1. fixed-I      |<I>_t / median(<I>) - 1| <= I_TOL. The method assumes I is
                  constant while Theta is scanned; the coldest setpoint
                  violates this in every sweep (I high by 6-35%).
  2. thermostat   max_t(<Theta>_t / T_t) <= 1, and <Theta>_t strictly
                  increasing in T_t. Otherwise the fit's x-axis is not the
                  controlled variable.
  3. barostat     std(P)/mean(P) <= P_TOL per setpoint (pressure scans only;
                  at P=2 the packing sits at the rigid-grain jamming point
                  and the barostat hunts, std/mean > 4).

Systematic error: a jackknife over SETPOINTS. Refit dropping each surviving
setpoint in turn; sigma_sys is the RMS deviation of those refits from the
full fit, scaled by sqrt(k-1) as for a standard jackknife. This directly
measures the window sensitivity that the statistical error omits, and it is
the number that must be quoted.

Cross-sweep comparisons must additionally use a COMMON window: call
fit_n(..., restrict_to=<set of Tgran>) with the intersection of the sweeps'
setpoints, otherwise the comparison inherits the systematic as a bias rather
than an error.
"""
import glob
import os
import re
import numpy as np

I_TOL = 0.03
P_TOL = 0.15


def parse_log(path):
    """Average the thermo block after the measurement marker; last 2/3 only."""
    try:
        L = open(path).readlines()
    except OSError:
        return None
    i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT WINDOW' in l), None)
    if i is None:
        return None
    h, rows = None, []
    for l in L[i:]:
        t = l.split()
        if t and t[0] == 'Step':
            h = t
            continue
        if h is None:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            if rows:
                break
            continue
        if len(v) == len(h):
            rows.append(v)
    if len(rows) < 4:
        return None
    a = np.array(rows)[len(rows) // 3:]
    c = {k: a[:, j] for j, k in enumerate(h)}
    out = dict(mu=float(c['v_muI'].mean()), Theta=float(c['v_Theta'].mean()),
               I=float(c['v_Iiner'].mean()))
    if 'v_P' in c:
        out['Pm'] = float(c['v_P'].mean())
        out['Ps'] = float(c['v_P'].std())
    if 'v_Zc' in c:
        out['Z'] = float(c['v_Zc'].mean())
    return out


def _slope(sel):
    th = np.array([r['Theta'] for r in sel])
    mu = np.array([r['mu'] for r in sel])
    m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return float(-c[0]), float(np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0]))


def surviving_setpoints(runs, i_tol=I_TOL, p_tol=P_TOL, restrict_to=None,
                        z_min=None):
    """Apply the gates; return the sorted list of surviving Tgran.

    z_min -- OPT-IN per-RUN jamming filter, added 2026-08-28. Drops individual
    runs whose measured Z is below the isostatic count, BEFORE any setpoint
    statistic is formed. Default None preserves the previous behaviour exactly.

    WHY THIS ONE GATE MAY BE PER-RUN, when the module header says gates never
    are. That rule exists because the I and thermostat gates are RELATIVE: they
    compare a setpoint against a global median, so applying them per run lets
    members of a deviant setpoint survive and can flip a fitted n negative. The
    jamming gate is not relative. Z >= D+1 is an ABSOLUTE physical statement --
    below it the packing is not rigid, and its "friction" is not the same
    quantity being fitted. That statement is
    true of a single run whatever its partners did, so per-run is the correct
    granularity and the setpoint mean was a coarsening.

    WHAT IT CATCHES. Surveyed over all 531 gated 2D+3D production runs, exactly
    4 are sub-isostatic while their setpoint mean still clears the gate -- all
    in 2D. Three are at the coldest setpoint Tgran = 5e-4, where least agitation
    leaves a pack able to stick in an under-coordinated metastable state; the
    fourth is a boundary case at Tgran = 3e-3 and is marked below. (The
    manuscript says three for this reason; do not "correct" it from an earlier
    version of this comment, which said all four.)

        mu_g = 0.1   seed 2   Z = 2.63 vs partners ~4.08   mu +19.1%
        mu_g = 0.15  seed 1   Z = 2.61 vs partners ~4.06   mu +26.5%
        mu_g = 1.0   seed 1   Z = 2.64 vs partners ~3.91   mu +24.0%
        mu_g = 1.0   seed 3   Z = 2.99 vs mean 3.00        mu  -1.5%  (boundary)

    Those runs sit at the coldest Theta -- the highest-leverage end of the
    fitted range -- so a single one distorts the whole curve. With three seeds
    and an adaptive fit they took mu_g = 0.1 to n = 0.041 +/- 0.175 and flipped
    mu_g = 1.0 negative. 3D has none.

    THIS IS CONDITIONING ON AN OUTCOME and must be reported as such: we keep
    runs that jammed and discard runs that did not. The justification is that
    the unjammed run is not a measurement of the target quantity at all, which
    is the same argument that justifies the gate existing in the first place.
    """
    byT = {}
    for r in runs:
        byT.setdefault(round(r['Tgran'], 12), []).append(r)
    if z_min is not None:
        # Per-run filter PLUS a majority rule, and the second half is not
        # optional. Filtering runs alone cherry-picks: at a setpoint straddling
        # the threshold it keeps whichever realisation landed a hair above and
        # admits a state the mean gate correctly rejected. Surveyed over all ten
        # sweeps, run-filtering alone removes 4 genuinely bad runs and wrongly
        # admits 10 such setpoints -- more harm than good. Computed by
        # analyze_jamming_gate.majority_rule_audit(); do not edit by hand.
        #
        # So: a setpoint survives only if a STRICT MAJORITY of its runs are
        # jammed, and it then carries just those runs. A minority surviving is
        # the unjamming signal the setpoint gate exists to catch, not a bad
        # realisation to prune. All 10 wrongly-admitted setpoints are 1-of-2 or
        # 1-of-3 within +/-0.02 of threshold; all 5 genuine outliers sit in
        # 3-run setpoints that keep their majority.
        byT = {t: keep for t, rs in byT.items()
               for keep in [[r for r in rs
                             if r.get('Z') is None or r['Z'] >= z_min]]
               if len(keep) * 2 > len(rs)}
    Ts = sorted(byT)
    if restrict_to is not None:
        keepset = {round(t, 12) for t in restrict_to}
        Ts = [t for t in Ts if t in keepset]
    if len(Ts) < 3:
        return [], byT
    # barostat
    if all('Ps' in r for t in Ts for r in byT[t]):
        Ts = [t for t in Ts
              if np.mean([r['Ps'] for r in byT[t]]) /
              max(np.mean([r['Pm'] for r in byT[t]]), 1e-30) <= p_tol]
    if len(Ts) < 3:
        return [], byT
    # fixed-I FIRST. Order matters: a setpoint can be both I-deviant and
    # Theta-anomalous, and if monotonicity is tested first the whole cell is
    # discarded instead of the one bad setpoint. Seen in 3D at mu_g=0.05 and
    # 0.1, where the coldest setpoint has Theta/Tgran = 0.41 and 0.75 against
    # ~0.04 elsewhere (thermostat not in control) AND I high by 7-22%. The I
    # gate removes it cleanly; testing monotonicity first threw away two of
    # the eight core 3D cells silently.
    Im = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
    med = np.median(list(Im.values()))
    Ts = [t for t in Ts if abs(Im[t] / med - 1) <= i_tol]
    if len(Ts) < 3:
        return [], byT
    # thermostat control, then monotonicity on what survives
    Th = {t: np.mean([r['Theta'] for r in byT[t]]) for t in Ts}
    Ts = [t for t in Ts if t > 0 and Th[t] / t <= 1.0]
    if len(Ts) < 3:
        return [], byT
    for a, b in zip(Ts, Ts[1:]):
        if Th[b] <= Th[a]:
            return [], byT
    return (Ts if len(Ts) >= 3 else []), byT


def fit_n(runs, i_tol=I_TOL, p_tol=P_TOL, restrict_to=None):
    """Return dict(n, stat, sys, tot, k, setpoints) or None.

    sys is a setpoint jackknife: the window sensitivity that the statistical
    error does not contain, and the reason two honest analyses of the same
    runs have disagreed by 2 sigma.
    """
    Ts, byT = surviving_setpoints(runs, i_tol, p_tol, restrict_to)
    if not Ts:
        return None
    sel = [r for t in Ts for r in byT[t]]
    n, stat = _slope(sel)
    if not np.isfinite(n):
        return None
    k = len(Ts)
    jk = []
    if k >= 4:
        for drop in Ts:
            s2 = [r for t in Ts if t != drop for r in byT[t]]
            nj, _ = _slope(s2)
            if np.isfinite(nj):
                jk.append(nj)
    sys_raw = (float(np.sqrt((k - 1) / k * np.sum((np.array(jk) - n) ** 2)))
               if len(jk) >= 3 else np.nan)
    # Same calibration as fit_local_sys: the raw jackknife has a non-zero null
    # expectation even when the model is exactly right, so quoting it as a
    # systematic inflates every error. Subtract the null in quadrature.
    sys_null = np.nan
    if np.isfinite(sys_raw):
        th = np.array([r['Theta'] for r in sel])
        mu = np.array([r['mu'] for r in sel])
        m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
        xx = np.log(th[m])
        grp = np.array([round(r['Tgran'], 12) for r in sel])[m]
        f = _jk_null_factor_A(np.vstack([xx, np.ones_like(xx)]).T, grp, 0)
        if np.isfinite(f):
            sys_null = f * stat
    if np.isfinite(sys_raw) and np.isfinite(sys_null):
        sys, ok = float(np.sqrt(max(sys_raw ** 2 - sys_null ** 2, 0.0))), True
    elif np.isfinite(sys_raw):
        sys, ok = sys_raw, True
    else:
        sys, ok = np.nan, False
    tot = float(np.hypot(stat, sys)) if ok else stat
    return dict(n=n, stat=stat, sys=sys, sys_raw=sys_raw, sys_null=sys_null,
                sys_ok=ok, tot=tot, k=k, setpoints=Ts, nruns=len(sel))


def load_sweep(pattern, label_rx, tgran_from=None):
    """Group logs into cells. label_rx must name groups; 'T' is the setpoint.

    tgran_from(match) -> actual Tgran, for sweeps whose label carries a
    reference setpoint that is rescaled at run time (the pressure scans use
    Tgran ~ P, so the label holds the P=10 value).
    """
    cells = {}
    for p in glob.glob(pattern):
        m = re.match(label_rx, os.path.basename(p))
        if not m:
            continue
        q = parse_log(p)
        if not q:
            continue
        q['Tgran'] = tgran_from(m) if tgran_from else float(m.group('T'))
        key = tuple((k, m.group(k)) for k in m.groupdict()
                    if k not in ('T', 's'))
        cells.setdefault(key, []).append(q)
    return cells


def fmt(res):
    if res is None:
        return 'VOID'
    s = f"{res['n']:+.4f} +/- {res['stat']:.4f}(stat)"
    if np.isfinite(res['sys']):
        s += f" +/- {res['sys']:.4f}(sys)  tot {res['tot']:.4f}"
    return s + f"  [{res['k']} setpoints]"


# LOCAL exponent at a reference temperature.
#
# mu(Theta) is not a power law -- the local slope rises with Theta (17/21
# cells, p=0.0072). A single n fitted over a window is therefore a
# window-average, and different windows legitimately disagree by more than
# the statistical error. That ambiguity is what sank most of the quantitative
# claims.
#
# The fix is not to remove the curvature (it is physical) but to stop
# averaging over it: fit a quadratic in log-log and quote the LOCAL slope at
# a stated reference temperature,
#
#     ln mu = a + b ln Theta + c (ln Theta)^2
#     n(Theta_0) = -(b + 2 c ln Theta_0)
#
# n(Theta_0) is unambiguous once Theta_0 is stated, and its error shrinks
# with data in the ordinary way. The curvature c is itself reported: it is a
# physical result, not a nuisance.
#
# Theta_0 MUST lie inside every cell being compared, or the comparison is an
# extrapolation. common_theta0() enforces that.

def fit_local(runs, theta0, i_tol=I_TOL, p_tol=P_TOL, restrict_to=None,
              gated_ts=None, min_k=4, z_min=None):
    """Quadratic log-log fit; return the local slope at theta0.

    Returns dict(n, stat, curv, curv_err, k, lo, hi) or None. 'lo'/'hi' are
    the Theta range actually fitted, so callers can verify theta0 is interior.

    gated_ts: skip gating and use exactly these setpoints. fit_local_sys uses
    it so that a jackknife replicate really is "the full sample minus one
    setpoint". Re-gating each replicate recomputes the median inside the
    fixed-I gate, which can silently drop a SECOND setpoint and turn the
    jackknife into a mixture of leave-one-out and leave-two-out.
    """
    if gated_ts is not None:
        Ts = list(gated_ts)
        byT = {}
        # z_min must be honoured here too: gated_ts freezes WHICH SETPOINTS are
        # used, not which runs within them. Skipping it would let a
        # sub-isostatic run back into every jackknife replicate. Uses the same
        # majority rule as surviving_setpoints, or a replicate would disagree
        # with the full fit about which setpoints exist.
        for r in runs:
            byT.setdefault(round(r['Tgran'], 12), []).append(r)
        if z_min is not None:
            byT = {t: keep for t, rs in byT.items()
                   for keep in [[r for r in rs
                                 if r.get('Z') is None or r['Z'] >= z_min]]
                   if len(keep) * 2 > len(rs)}
        Ts = [t for t in Ts if t in byT]
    else:
        Ts, byT = surviving_setpoints(runs, i_tol, p_tol, restrict_to, z_min)
    if len(Ts) < min_k:                  # need >=4 for a meaningful quadratic
        return None
    sel = [r for t in Ts for r in byT[t]]
    th = np.array([r['Theta'] for r in sel])
    mu = np.array([r['mu'] for r in sel])
    m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
    if m.sum() < 5:
        return None
    x = np.log(th[m]) - np.log(theta0)   # centre on theta0 so b IS the slope
    y = np.log(mu[m])
    A = np.vstack([np.ones_like(x), x, x ** 2]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = max(m.sum() - 3, 1)
    s2 = float(np.sum(r ** 2) / dof)
    cov = s2 * np.linalg.inv(A.T @ A)
    return dict(n=float(-c[1]), stat=float(np.sqrt(cov[1, 1])),
                curv=float(-2 * c[2]), curv_err=float(2 * np.sqrt(cov[2, 2])),
                k=len(Ts), lo=float(th[m].min()), hi=float(th[m].max()),
                theta0=float(theta0))


# ADAPTIVE degree. The quadratic was introduced because a straight line in
# log-log is inadequate -- mu(Theta) is curved, so a window-averaged n depends
# on the window. That same argument does not stop at second order, and it was
# checked: a cubic term is significant at p < 0.003 in 3D at every mu_g >= 0.2
# (d = -0.03 to -0.05) and in 2D across the crossing band mu_g = 0.06-0.13
# (d = +0.02 to +0.03, p < 0.06). Where a cubic is needed, the quadratic's
# curvature is itself a window average -- which is exactly why the curvature
# zero-crossing moved from 0.103 to 0.145 when the fit was restricted to the
# upper half of the Theta range.
#
# So the degree is chosen per cell by a sequential F-test rather than fixed.
# n(Theta_0) = -b is unchanged in meaning; it is just no longer contaminated by
# unmodelled higher-order structure. Where the quadratic is adequate this
# returns exactly the quadratic, so nothing that was already sound moves.

def _poly_fit(x, y, deg):
    A = np.vander(x, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c, float(r @ r), A


def fit_local_adaptive(runs, theta0, max_deg=4, alpha=0.05,
                       i_tol=I_TOL, p_tol=P_TOL, restrict_to=None,
                       gated_ts=None, min_k=4, force_deg=None, z_min=None):
    """Local slope at theta0 with the polynomial degree chosen by F-test.

    Returns dict(n, stat, curv, curv_err, deg, k, lo, hi, theta0) or None.
    'deg' is the degree actually selected, and must be reported: a result that
    only holds at one fixed degree is a result about the fit, not the physics.

    NOTE: fit_local_sys cannot calibrate the jackknife for this estimator (the
    null factor assumes a fixed quadratic design), so it falls back to the raw,
    uncalibrated jackknife here. That is conservative, and it is why the
    adaptive fit stays a cross-check rather than the reported number.
    """
    from scipy import stats

    if gated_ts is not None:
        byT = {}
        for r in runs:                      # see fit_local: z_min applies here too
            byT.setdefault(round(r['Tgran'], 12), []).append(r)
        if z_min is not None:
            byT = {t: keep for t, rs in byT.items()
                   for keep in [[r for r in rs
                                 if r.get('Z') is None or r['Z'] >= z_min]]
                   if len(keep) * 2 > len(rs)}
        Ts = [t for t in gated_ts if t in byT]
    else:
        Ts, byT = surviving_setpoints(runs, i_tol, p_tol, restrict_to, z_min)
    if len(Ts) < min_k:
        return None
    sel = [r for t in Ts for r in byT[t]]
    th = np.array([r['Theta'] for r in sel])
    mu = np.array([r['mu'] for r in sel])
    m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
    n_pts = int(m.sum())
    if n_pts < 5:
        return None
    x = np.log(th[m]) - np.log(theta0)
    y = np.log(mu[m])

    # force_deg: skip selection and fit exactly this degree. Used by
    # fit_local_sys to FREEZE the degree across jackknife replicates. Without
    # it each leave-one-out refit re-selects its own degree -- measured on the
    # real cells, 8 of 16 flip -- so the jackknife spread mixes a window
    # systematic with a model-class change, and no fixed-design null describes
    # it. Freezing makes the design constant across replicates, which is the
    # condition _jk_null_factor_A assumes; the cost of selection is then quoted
    # separately as a degree systematic rather than silently folded in.
    if force_deg is not None:
        deg = int(force_deg)
        if len(Ts) < deg + 1 or n_pts < deg + 3:
            return None
        c, rss, A = _poly_fit(x, y, deg)
        dof = max(n_pts - (deg + 1), 1)
        cov = (rss / dof) * np.linalg.inv(A.T @ A)
        return dict(n=float(-c[1]), stat=float(np.sqrt(cov[1, 1])),
                    curv=float(-2 * c[2]), curv_err=float(2 * np.sqrt(cov[2, 2])),
                    deg=deg, k=len(Ts), lo=float(th[m].min()),
                    hi=float(th[m].max()), theta0=float(theta0), forced=True)

    deg = 2
    c, rss, A = _poly_fit(x, y, deg)
    while deg < max_deg:
        # a degree needs both enough setpoints to constrain it and enough
        # residual dof to test it
        if len(Ts) < deg + 3 or n_pts - (deg + 2) < 3:
            break
        c2, rss2, A2 = _poly_fit(x, y, deg + 1)
        dof2 = n_pts - (deg + 2)
        if rss2 <= 0:
            break
        F = (rss - rss2) / (rss2 / dof2)
        if 1 - stats.f.cdf(F, 1, dof2) >= alpha:
            break
        deg, c, rss, A = deg + 1, c2, rss2, A2

    dof = max(n_pts - (deg + 1), 1)
    cov = (rss / dof) * np.linalg.inv(A.T @ A)
    return dict(n=float(-c[1]), stat=float(np.sqrt(cov[1, 1])),
                curv=float(-2 * c[2]), curv_err=float(2 * np.sqrt(cov[2, 2])),
                deg=deg, k=len(Ts), lo=float(th[m].min()), hi=float(th[m].max()),
                theta0=float(theta0))


def _jk_null_factor_A(A, groups, idx):
    """E[sigma_jk]/sigma_stat for coefficient `idx` of design `A`. See below."""
    try:
        G = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return np.nan
    p = A.shape[1]
    c = G[idx] @ A.T
    keys = sorted(set(groups))
    k = len(keys)
    if k < p + 1:
        return np.nan
    acc = 0.0
    for g in keys:
        m = groups != g
        Ag = A[m]
        if Ag.shape[0] < p + 1 or np.linalg.matrix_rank(Ag) < p:
            return np.nan
        cg = np.zeros_like(c)
        cg[m] = np.linalg.inv(Ag.T @ Ag)[idx] @ Ag.T
        acc += float(np.sum((cg - c) ** 2))
    stat_unit = float(np.sqrt(G[idx, idx]))
    if stat_unit <= 0:
        return np.nan
    return float(np.sqrt((k - 1) / k * acc) / stat_unit)


def _jk_null_factor(x, groups):
    """E[sigma_jk] / sigma_stat for a quadratic slope, under a TRUE model.

    The setpoint jackknife is not a systematic. It is a variance estimator, and
    on data with no window systematic at all it still returns something -- about
    1.3x the OLS error for these designs. Adding that in quadrature to stat
    inflates every error bar by ~1.6x for nothing, and (worse) makes null
    results and self-consistency checks look better than they are.

    The null expectation is exactly computable, no Monte Carlo needed. Both the
    full fit and each leave-one-setpoint-out refit are LINEAR in y:

        n = -c.y            n_(g) = -c_g.y

    and both are unbiased for the slope when the quadratic is true, so
    (c_g - c).X.beta = 0 and the difference is pure noise:

        E[sigma_jk^2] = (k-1)/k * sigma^2 * sum_g ||c_g - c||^2

    Returning that as a ratio to sigma_stat makes it a pure property of the
    DESIGN (which setpoints, how many seeds at each) -- independent of the
    noise level, and deterministic.
    """
    return _jk_null_factor_deg(x, groups, 2)


def _jk_null_factor_deg(x, groups, deg):
    """The same null, for a polynomial design of arbitrary degree.

    _jk_null_factor hardcodes the quadratic. The adaptive estimator may sit at
    degree 3 or 4, and its jackknife then has a different null -- higher, since
    more parameters means the leave-one-out coefficient vectors move further.
    Using the quadratic null there would under-subtract and overstate the
    systematic; using no null at all (the previous behaviour) over-subtracts
    nothing and overstates it far more.
    """
    A = np.vander(np.asarray(x, float), deg + 1, increasing=True)
    return _jk_null_factor_A(A, groups, 1)


def fit_local_sys(runs, theta0, fn=None, calibrate=True, **kw):
    """fit_local plus a CALIBRATED setpoint-jackknife systematic on n(theta_0).

    WHY A SYSTEMATIC IS NEEDED. n(theta_0) has no window ambiguity only if the
    fitted model is ADEQUATE over the window, and it is not everywhere: in 3D a
    cubic is required at every mu_g >= 0.2. Quoting stat alone there understates
    the error.

    WHY IT MUST BE CALIBRATED. The raw jackknife has a large null expectation
    (see _jk_null_factor) -- roughly 1.3 x stat on these designs -- so a raw
    sys/stat near 1 means nothing was found. Measured against its own null:
    seven of the eight core 2D cells fall BELOW it (there is no detectable
    window systematic in 2D at all), while 3D above mu_g = 0.1 shows a real
    excess of 0.009-0.036. Quoting the raw jackknife inflated every 2D error by
    up to 1.6x, which weakened the positive claims and flattered the nulls.

    So the reported systematic is the EXCESS over the null:

        sys = sqrt(max(sys_raw^2 - sys_null^2, 0))

    Fields: sys_raw (what the old code returned), sys_null (the noise floor),
    sys (the excess, quoted), tot = hypot(stat, sys), and sys_ok -- False when
    no systematic could be formed, which callers MUST check rather than
    silently receiving tot == stat.
    """
    fn = fn or fit_local
    full = fn(runs, theta0, **kw)
    if full is None:
        return None
    # If the caller froze the setpoint list itself, the jackknife must be taken
    # over THAT list, not over a freshly-gated one -- and gated_ts must be
    # stripped from kw2 or it is supplied twice below (TypeError).
    if kw.get('gated_ts') is not None:
        byT = {}
        for r in runs:
            byT.setdefault(round(r['Tgran'], 12), []).append(r)
        Ts = [t for t in kw['gated_ts'] if t in byT]
    else:
        Ts, byT = surviving_setpoints(runs, **{k: v for k, v in kw.items()
                                               if k in ('i_tol', 'p_tol',
                                                        'restrict_to', 'z_min')})
    kw2 = {k: v for k, v in kw.items()
           if k not in ('restrict_to', 'gated_ts')}

    # FREEZE THE DEGREE across replicates (adaptive estimator only). Each
    # leave-one-out refit would otherwise re-select its own degree, and 8 of
    # 16 real cells flip -- so sigma_jk would mix a window systematic with a
    # model-class change and no fixed-design null would describe it. Frozen,
    # the design is constant across replicates, which is the condition
    # _jk_null_factor_A assumes. The cost of selection is then reported
    # separately as sys_deg rather than silently folded in.
    deg = full.get('deg')
    frozen = (fn is fit_local_adaptive) and deg is not None
    rep_kw = dict(kw2, force_deg=deg) if frozen else kw2

    vals = []
    for drop in Ts:
        # gated_ts: leave-one-out on the FROZEN gate decision (see fit_local)
        r = fn(runs, theta0, gated_ts=[t for t in Ts if t != drop],
               min_k=3, **rep_kw)
        if r:
            vals.append(r['n'])
    k = len(vals)
    sys_raw = (float(np.sqrt((k - 1) / k * np.sum((np.array(vals) - full['n']) ** 2)))
               if k >= 3 else np.nan)

    sys_null = np.nan
    if calibrate and (fn is fit_local or frozen) and np.isfinite(sys_raw):
        sel = [r for t in Ts for r in byT[t]]
        th = np.array([r['Theta'] for r in sel])
        mu = np.array([r['mu'] for r in sel])
        m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
        x = np.log(th[m]) - np.log(theta0)
        grp = np.array([round(r['Tgran'], 12) for r in sel])[m]
        f = _jk_null_factor_deg(x, grp, deg if frozen else 2)
        if np.isfinite(f):
            sys_null = f * full['stat']

    if np.isfinite(sys_raw) and np.isfinite(sys_null):
        sys = float(np.sqrt(max(sys_raw ** 2 - sys_null ** 2, 0.0)))
        ok = True
    elif np.isfinite(sys_raw):
        sys, ok = sys_raw, True          # uncalibrated (adaptive fit): conservative
    else:
        sys, ok = np.nan, False

    # THE DEGREE SYSTEMATIC, quoted separately rather than folded in. The
    # audit found model selection moves n by 1.7-3.7x the quoted error in five
    # of eight 3D cells, and that this sits outside the error budget. It now
    # has a number: the largest shift in n from refitting the same data at the
    # neighbouring degrees.
    sys_deg = 0.0
    if frozen:
        shifts = []
        for d in (deg - 1, deg + 1):
            if d < 2:
                continue
            r = fn(runs, theta0, gated_ts=list(Ts), min_k=3,
                   **dict(kw2, force_deg=d))
            if r:
                shifts.append(abs(r['n'] - full['n']))
        sys_deg = float(max(shifts)) if shifts else 0.0

    full = dict(full)
    full.update(sys=sys, sys_raw=sys_raw, sys_null=sys_null, sys_ok=ok,
                sys_deg=sys_deg,
                tot=float(np.hypot(full['stat'], sys)) if ok else full['stat'])
    return full


def common_theta0(cell_runs, **kw):
    """Geometric centre of the Theta range shared by every cell given."""
    los, his = [], []
    for runs in cell_runs:
        Ts, byT = surviving_setpoints(runs, **kw)
        if len(Ts) < 4:
            continue
        th = [r['Theta'] for t in Ts for r in byT[t] if r['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    if not los:
        return None
    lo, hi = max(los), min(his)          # the INTERSECTION
    if hi <= lo:
        return None
    return float(np.sqrt(lo * hi)), (lo, hi)
