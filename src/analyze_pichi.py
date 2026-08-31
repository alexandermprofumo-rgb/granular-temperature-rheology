"""
THE THIRD LEVER ON chi: does n collapse onto chi under a pressure lever?

chi (the Coulomb-mobilized contact fraction) is the only microstructural
candidate not falsified once the window systematic is carried -- 1.4 sigma,
against 2.7-3.5 for the other eight. But that test only has power against
discrepancies above 0.065 in n over an n-span of 0.143, so "not falsified" is
partly just lost power. Pressure is orthogonal to both friction and stiffness
and spans 10x here.

DESIGN CHOICE THAT MATTERS. The reference curve n(chi) is built from the NEW
P = 5 runs, not from the existing P = 10 data, even though the latter exists.
The P = 10 chi values in tier1_matched2d_results.csv come from a different
extraction pipeline; using them as the reference would confound a pipeline
difference with the physics being tested. Everything here goes through one
extractor, applied identically to both pressures. The P = 10 data is reported
alongside as context, never as part of the test.

The test is symmetric and run both ways: build n(chi) by varying friction at
one pressure, then ask whether the other pressure's points -- reached by a
route that never touches friction -- land on it.

CHI EXTRACTION. chi = fraction of REAL contacts with |f_t| >= 0.99 mu_g |f_n|.
Two things to get right, both verified against the dumps:
  * the normal force is columns 6:8. Columns 4:6 are the separation vector,
    and reading those as force silently turns the threshold into mu_g*|d| ~
    mu_g, which is meaningless.
  * ~25% of dump entries are neighbour pairs with ZERO force -- inside the
    neighbour cutoff but not touching. They must be filtered or chi is
    diluted by a quarter.
LAMMPS clamps |f_t| <= mu_g |f_n| exactly, so mobilized contacts sit at a
ratio of 1.0000 and the 0.99 threshold is not a tuning knob.

Usage:  python3 analyze_pichi.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit

CACHE = 'pichi_chi.csv'
MOB = 0.99


# chi extraction

def frames(path):
    """Yield per-frame arrays from a LAMMPS local dump."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                n = int(f.readline())
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                rows = []
                for _ in range(n):
                    t = f.readline().split()
                    if len(t) >= 11:
                        rows.append([float(x) for x in t[:11]])
                if rows:
                    yield np.array(rows)


def chi_of(path, mu_g):
    """Mobilized fraction, averaged over frames; also mean Z-proxy."""
    vals, ncon = [], []
    for a in frames(path):
        fn = np.linalg.norm(a[:, 6:8], axis=1)     # NORMAL force, not 4:6
        live = fn > 0                               # drop untouching neighbours
        if live.sum() < 100:
            continue
        ratio = a[live, 10] / (mu_g * fn[live])
        vals.append(float((ratio >= MOB).mean()))
        ncon.append(int(live.sum()))
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), int(np.mean(ncon)), len(vals)


def build_cache(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('P', 'mu_g', 'Tgran', 'chi', 'chi_sd', 'ncon', 'nframe'):
                r[k] = float(r[k])
        return rows
    rx = re.compile(r'dump\.contacts\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                    r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    rows = []
    paths = sorted(glob.glob('sweep_pichi/dump.contacts.P*'))
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        P = float(m.group('P')); mg = float(m.group('mu'))
        got = chi_of(p, mg)
        if not got:
            continue
        chi, sd, nc, nf = got
        rows.append(dict(P=P, mu_g=mg, Tgran=float(m.group('T')) * P / 10.0,
                         seed=m.group('s'), chi=chi, chi_sd=sd, ncon=nc, nframe=nf))
        if (i + 1) % 25 == 0:
            print(f'   ... {i+1}/{len(paths)} dumps', file=sys.stderr)
    if rows:
        with open(CACHE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


# n and the collapse test

def run_thetas():
    """Measured Theta per individual run, keyed (P, mu_g, Tgran, seed).

    Needed because chi must be evaluated AT Theta_0, not averaged over the
    cell. chi varies across the Theta window by nearly as much as it varies
    across frictions (within-cell sd 0.030 against across-cell spread 0.056),
    so pairing a window-averaged chi with a local n(Theta_0) compares two
    quantities defined at different temperatures -- the same class of error as
    the window-averaged n this project already had to abandon.
    """
    rx = re.compile(r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                    r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    out = {}
    for p in glob.glob('sweep_pichi/log.P*'):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        P = float(m.group('P'))
        key = (P, float(m.group('mu')),
               round(float(m.group('T')) * P / 10.0, 12), m.group('s'))
        out[key] = q['Theta']
    return out


def chi_at(rows, thetas, P, mg, theta0, runs):
    """chi interpolated to theta0, using only gated setpoints.

    Returns (chi, err) or None. A straight line in log chi vs log Theta is
    enough over the ~10x window and keeps the extrapolation honest; the error
    is propagated from the seed scatter.
    """
    Ts, _ = nfit.surviving_setpoints(runs)
    keep = {round(t, 12) for t in Ts}
    xs, ys = [], []
    for r in rows:
        if r['P'] != P or r['mu_g'] != mg:
            continue
        tg = round(r['Tgran'], 12)
        if tg not in keep:
            continue
        th = thetas.get((P, mg, tg, str(r['seed'])))
        if th is None or th <= 0 or r['chi'] <= 0:
            continue
        xs.append(np.log(th)); ys.append(np.log(r['chi']))
    if len(xs) < 4:
        return None
    xs = np.array(xs); ys = np.array(ys)
    x0 = np.log(theta0)
    A = np.vstack([np.ones_like(xs), xs - x0]).T
    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r = ys - A @ c
    dof = max(len(xs) - 2, 1)
    cov = (float(r @ r) / dof) * np.linalg.inv(A.T @ A)
    chi = float(np.exp(c[0]))
    return chi, float(chi * np.sqrt(cov[0, 0]))


def load_n():
    raw = nfit.load_sweep(
        'sweep_pichi/log.P*',
        r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
        tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    out = {}
    for key, runs in raw.items():
        d = dict(key)
        out[(float(d['P']), float(d['mu']))] = runs
    return out


def reduced_theta0(cellmap, P, mus):
    los, his = [], []
    for m in mus:
        runs = cellmap.get((P, m))
        if not runs:
            continue
        Ts, byT = nfit.surviving_setpoints(runs)
        if len(Ts) < 4:
            continue
        th = [r['Theta'] / P for t in Ts for r in byT[t] if r['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    if not los:
        return None
    lo, hi = max(los), min(his)
    return float(np.sqrt(lo * hi)) if hi > lo else None


# The offset, measured properly.
#
# Adding the two curves' systematics in quadrature DOUBLE-COUNTS a common-mode
# effect. Both curves are built with the same Theta_0 construction from the
# same setpoint grid, so a window choice that pushes the P = 5 curve up pushes
# the P = 50 curve up too, and the OFFSET between them is far less sensitive to
# it than either curve alone. The independent-sum error (~0.032) is therefore an
# overestimate, and it is what pins the test at ~1.4 sigma.
#
# The correct systematic on the offset is a PAIRED jackknife: drop one setpoint
# from BOTH pressures at once, recompute everything downstream, and take the
# spread of the resulting offsets.
#
# GUARD. A jackknife error can also collapse for uninteresting reasons -- if
# the offset is insensitive to the setpoint grid because it is pinned by
# something else entirely. So the paired jackknife is cross-checked against a
# fully independent SEED SPLIT (seeds 1,2 against 3,4), which shares no
# machinery with it. If the two disagree badly, the small error is an artefact
# and must not be used.

def per_run():
    """(P, mu_g, T10, seed) -> parsed thermo record. Keeps the seed, which
    nfit.load_sweep drops, so the seed-split cross-check is possible."""
    rx = re.compile(r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                    r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    out = {}
    for p in glob.glob('sweep_pichi/log.P*'):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        P = float(m.group('P'))
        q['Tgran'] = float(m.group('T')) * P / 10.0
        out[(P, float(m.group('mu')), m.group('T'), m.group('s'))] = q
    return out


def offset_once(runs_by, rows, thetas, Ps, mus, t10s, seeds):
    """The coherent offset between the two pressures, for one subset of the
    setpoint grid and seeds. Returns (offset, npts) or None."""
    cell = {}
    for (P, mg, t10, s), q in runs_by.items():
        if t10 not in t10s or s not in seeds:
            continue
        cell.setdefault((P, mg), []).append(q)
    # ONE reduced Theta_0 for BOTH pressures. Taking the intersection per
    # pressure instead put P = 5 at reduced 2.01e-4 and P = 50 at 1.54e-4, a
    # 30% mismatch -- and since n rises with Theta that alone lifts the P = 5
    # curve above the P = 50 one, which is the sign of the offset being
    # measured. It accounted for about 20% of it.
    los, his = [], []
    for P in Ps:
        for mg in mus:
            r = cell.get((P, mg))
            if not r:
                continue
            Ts, byT = nfit.surviving_setpoints(r)
            if len(Ts) < 4:
                continue
            th = [x['Theta'] / P for t in Ts for x in byT[t] if x['Theta'] > 0]
            if th:
                los.append(min(th)); his.append(max(th))
    if not los or max(los) >= min(his):
        return None
    T0h = float(np.sqrt(max(los) * min(his)))

    pts = {}
    for P in Ps:
        T0 = T0h * P
        for mg in mus:
            r = cell.get((P, mg))
            if not r:
                continue
            f = nfit.fit_local(r, T0)
            cx = chi_at(rows, thetas, P, mg, T0, r)
            if f and cx:
                pts.setdefault(P, []).append((cx[0], f['n'], mg))
    if len(pts) < 2:
        return None
    ref, test = min(pts), max(pts)
    A = sorted(pts[ref]); B = sorted(pts[test])
    fx = np.array([p[0] for p in A]); fy = np.array([p[1] for p in A])
    d = [n - float(np.interp(c, fx, fy)) for c, n, _ in B
         if fx.min() <= c <= fx.max()]
    if len(d) < 3:
        return None
    return float(np.mean(d)), len(d)


def paired_offset(rows, thetas):
    runs_by = per_run()
    Ps = sorted({k[0] for k in runs_by})
    mus = sorted({k[1] for k in runs_by})
    t10s = sorted({k[2] for k in runs_by}, key=float)
    seeds = sorted({k[3] for k in runs_by})
    print()
    print('=' * 76)
    print('THE OFFSET, WITH A PAIRED JACKKNIFE')
    print('=' * 76)
    full = offset_once(runs_by, rows, thetas, Ps, mus, set(t10s), set(seeds))
    if not full:
        print('   cannot form the full-sample offset')
        return
    print(f'   full sample: offset = {full[0]:+.4f}  ({full[1]} overlapping points, '
          f'{len(t10s)} setpoints, {len(seeds)} seeds)')

    vals = []
    for drop in t10s:
        r = offset_once(runs_by, rows, thetas, Ps, mus,
                        {t for t in t10s if t != drop}, set(seeds))
        if r:
            vals.append(r[0])
    if len(vals) >= 3:
        k = len(vals)
        jk = float(np.sqrt((k - 1) / k * np.sum((np.array(vals) - full[0]) ** 2)))
        print(f'   paired jackknife over setpoints: {jk:.4f}   '
              f'(offset ranges {min(vals):+.4f} to {max(vals):+.4f} over {k} drops)')
    else:
        jk = float('nan')
        print('   paired jackknife: too few usable replicates')

    # independent cross-check: does a seed split give a compatible error?
    half = len(seeds) // 2
    if half >= 1 and len(seeds) >= 2:
        a = offset_once(runs_by, rows, thetas, Ps, mus, set(t10s), set(seeds[:half]))
        b = offset_once(runs_by, rows, thetas, Ps, mus, set(t10s), set(seeds[half:]))
        if a and b:
            spread = abs(a[0] - b[0]) / 2.0
            print(f'   seed split: seeds {seeds[:half]} -> {a[0]:+.4f}, '
                  f'{seeds[half:]} -> {b[0]:+.4f}   half-difference {spread:.4f}')
            print(f'   -> independent estimate of the statistical error on the '
                  f'offset: ~{spread/np.sqrt(2):.4f}')
            if np.isfinite(jk) and jk > 0:
                ratio = spread / np.sqrt(2) / jk
                print(f'   -> seed-split / paired-jackknife = {ratio:.1f}x')
                # ONE-SIDED. The seed split measures the STATISTICAL error; the
                # paired jackknife measures the WINDOW SYSTEMATIC. They are
                # different quantities, so seed-split < jackknife is the normal
                # case (the systematic dominates, as it does throughout this
                # project) and is not a warning. Only the reverse matters: if
                # seed-to-seed scatter EXCEEDS the jackknife, the jackknife is
                # missing real variability and its error must not be quoted.
                if ratio > 3:
                    print('      WARNING: seed scatter exceeds the paired jackknife by')
                    print('      more than 3x -- the jackknife is missing real')
                    print('      variability. Do NOT quote it.')
    if np.isfinite(jk) and jk > 0:
        print(f'\n   offset = {full[0]:+.4f} +/- {jk:.4f} (paired) '
              f'-> {abs(full[0])/jk:.1f} sigma')
        print('   Read this ONLY if the seed split above agrees with the paired')
        print('   jackknife; otherwise the systematic is understated.')


def main():
    print('extracting chi from contact dumps ...', file=sys.stderr)
    rows = build_cache('--recache' in sys.argv)
    if not rows:
        print('no contact dumps found yet')
        return
    cellmap = load_n()
    Ps = sorted({r['P'] for r in rows})
    mus = sorted({r['mu_g'] for r in rows})
    print(f'chi extracted from {len(rows)} runs; pressures {Ps}, '
          f'frictions {[f"{m:g}" for m in mus]}\n')

    # how much does chi move within a cell (across Theta)?
    print('=' * 76)
    print('chi(mu_g, P), and its variation across the Theta window')
    print('=' * 76)
    print(f'{"P":>6}{"mu_g":>7}{"chi":>9}{"sd over Theta":>15}{"n cells":>9}')
    chimap = {}
    for P in Ps:
        for mg in mus:
            s = [r for r in rows if r['P'] == P and r['mu_g'] == mg]
            if not s:
                continue
            c = np.array([r['chi'] for r in s])
            chimap[(P, mg)] = (float(c.mean()), float(c.std()))
            print(f'{P:>6g}{mg:>7g}{c.mean():>9.4f}{c.std():>15.4f}{len(s):>9}')
    print('\n   If sd over Theta is comparable to the spread of chi ACROSS')
    print('   frictions, chi is not a cell property and the test below is')
    print('   ill-posed -- check before reading the collapse result.')
    if chimap:
        across = np.std([v[0] for v in chimap.values()])
        within = np.mean([v[1] for v in chimap.values()])
        print(f'   spread across cells {across:.4f}  vs  mean within-cell '
              f'{within:.4f}   ratio {across/max(within,1e-9):.1f}x')

    # n per cell
    print()
    print('=' * 76)
    print('n(Theta_0) per cell, and the collapse test')
    print('=' * 76)
    thetas = run_thetas()
    pts = {}
    for P in Ps:
        T0h = reduced_theta0(cellmap, P, mus)
        if T0h is None:
            print(f'   P = {P:g}: no common reduced-Theta range yet')
            continue
        T0 = T0h * P
        for mg in mus:
            runs = cellmap.get((P, mg))
            if not runs:
                continue
            r = nfit.fit_local_sys(runs, T0)
            cx = chi_at(rows, thetas, P, mg, T0, runs)
            if r and cx:
                pts.setdefault(P, []).append((cx[0], r['n'], r['tot'], mg, cx[1]))
        if P in pts:
            print(f'\n   P = {P:g}  (That_0 = {T0h:.2e}, Theta_0 = {T0:.2e})')
            for c, n, e, mg, ce in sorted(pts[P]):
                print(f'      mu_g={mg:<5g} chi(Theta_0)={c:.4f} +/- {ce:.4f}'
                      f'   n={n:+.4f} +/- {e:.4f}')

    if len(pts) < 2:
        print('\n   need both pressures fitted for the collapse test')
        return

    print()
    print('=' * 76)
    print('COLLAPSE TEST -- do the two pressures lie on one n(chi) curve?')
    print('=' * 76)
    lo, hi = min(pts), max(pts)
    for ref, test in ((lo, hi), (hi, lo)):
        A = sorted(pts[ref]); B = sorted(pts[test])
        fx = np.array([p[0] for p in A]); fy = np.array([p[1] for p in A])
        fe = np.array([p[2] for p in A])
        # the reference curve's slope converts the chi error into an n error;
        # ignoring it would understate the uncertainty wherever n(chi) is steep
        slope = np.gradient(fy, fx) if len(fx) > 2 else np.zeros(len(fx))
        sig, errs = [], []
        for c, n, e, mg, ce in B:
            if c < fx.min() or c > fx.max():
                continue
            pred = float(np.interp(c, fx, fy))
            perr = float(np.interp(c, fx, fe))
            sl = float(np.interp(c, fx, slope))
            tot = float(np.sqrt(e ** 2 + perr ** 2 + (sl * ce) ** 2))
            sig.append(((n - pred) / tot, mg, n, pred))
            errs.append(tot)
        if len(sig) < 3:
            print(f'   reference P={ref:g}: only {len(sig)} of {len(B)} test points '
                  f'fall inside the reference chi range -- no overlap, no test')
            continue
        rms = float(np.sqrt(np.mean([s[0] ** 2 for s in sig])))
        print(f'\n   reference curve P = {ref:g}, testing P = {test:g}  '
              f'({len(sig)} overlapping points)')
        for s, mg, n, pred in sig:
            print(f'      mu_g={mg:<5g} n={n:+.4f} predicted {pred:+.4f}   {s:+.1f} sigma')
        print(f'      RMS = {rms:.1f} sigma  -> '
              f'{"FAILS -- chi is not a state function" if rms >= 2 else "no failure detected"}')

        # COHERENT OFFSET. RMS treats a uniform shift between the two levers as
        # if it were scatter, and throws away the one thing a state-function
        # violation actually looks like: every test point landing on the SAME
        # side of the reference curve. In pass 1 all four residuals were
        # negative, which RMS scored as an unremarkable 1.3 sigma.
        #
        # The error on the mean does NOT fall as 1/sqrt(N). The reference
        # curve's own systematic is common-mode -- it shifts every comparison
        # together -- so only the test points' independent errors average down:
        #     sigma_mean^2 = <sigma_indep^2>/N + sigma_ref_sys^2
        # Treating it as sigma/sqrt(N) would manufacture significance, which is
        # this project's most-repeated failure mode.
        d = np.array([s[2] - s[3] for s in sig])
        ref_sys = float(np.mean(fe))          # common-mode, does not average
        indep = float(np.mean([p[2] for p in B if p[0] >= fx.min()
                               and p[0] <= fx.max()]))
        se = float(np.sqrt(indep ** 2 / len(d) + ref_sys ** 2))
        mo = float(np.mean(d))
        nneg = int((d < 0).sum())
        print(f'      coherent offset: {mo:+.4f} +/- {se:.4f} -> {abs(mo)/se:.1f} sigma'
              f'   ({nneg}/{len(d)} negative)')
        if abs(mo) / se >= 2:
            print('      -> the two levers are OFFSET: chi is not a state function')

        # POWER. Without this the verdict is uninterpretable: a test that
        # cannot detect anything always returns "no failure". Cf. the
        # under-powered curvature F-test that misdirected this project.
        span = float(fy.max() - fy.min())
        print(f'      power: mean error on the discrepancy {np.mean(errs):.4f}, '
              f'so 2-sigma detectable = {2*np.mean(errs):.4f}, '
              f'against an n-span of {span:.4f}')
        if 2 * np.mean(errs) > 0.5 * span:
            print('      -> UNDER-POWERED: the detectable discrepancy exceeds half')
            print('         the range being probed. "No failure" here is not evidence')
            print('         of a collapse.')

    paired_offset(rows, thetas)


if __name__ == '__main__':
    main()
