"""
Do the anisotropy LEVELS work as state variables for n?

Nine microstructural candidates have been falsified. The channel decomposition
(§5) says why: n = C_c + C_n + C_t is a sum of three channel DERIVATIVES, so a
single variable can only organize it if the channels are slaved together.

That argument makes a sharp prediction worth testing rather than assuming. The
anisotropies a_c, a_n, a_t are now known to be the *right* variables in the
sense that they reconstruct mu exactly — so if any single static quantity were
going to be a state function for n, these are the strongest candidates
available, far better motivated than the nine. The decomposition predicts they
still fail, because n depends on their derivatives, not their levels.

This is a genuine test of the framework: if a_n (say) DID collapse n across
levers, the decomposition's central claim would be wrong.

Method is identical to the chi test (§7a) so the numbers are comparable:
friction lever at one pressure builds the reference curve n(X), the other
pressure supplies test points reached without touching friction, and the
statistic is the COHERENT OFFSET with a paired jackknife -- RMS scores a
uniform shift as unremarkable scatter.

Everything is measured, not closed: a_t comes from the 11-column pichi dumps.

Usage:  python3 analyze_anis_levels.py
"""
import numpy as np

import nfit
from analyze_pichi import load_n, reduced_theta0
from analyze_rb import local
import csv


def load_rows():
    rows = list(csv.DictReader(open('rb_channels.csv')))
    for r in rows:
        for k in ('P', 'mu_g', 'Tgran', 'mu', 'a_c', 'a_n', 'a_t', 'Theta'):
            r[k] = float(r[k])
    return rows


def level_at(rows, cm, P, mg, T0, var):
    """The anisotropy LEVEL at Theta_0, on gated setpoints."""
    rr = cm.get((P, mg))
    if not rr:
        return None
    Ts, _ = nfit.surviving_setpoints(rr)
    keep = {round(t, 12) for t in Ts}
    sel = [r for r in rows if r['P'] == P and r['mu_g'] == mg
           and round(r['Tgran'], 12) in keep and r['Theta'] > 0]
    if len(sel) < 8:
        return None
    g = local(np.log([r['Theta'] for r in sel]), [r[var] for r in sel], np.log(T0))
    return None if g is None else g[0]


def main():
    rows = load_rows()
    cm = load_n()
    mus = sorted({r['mu_g'] for r in rows})
    Ps = sorted({r['P'] for r in rows})

    los, his = [], []
    for (P, mg), rr in cm.items():
        Ts, byT = nfit.surviving_setpoints(rr)
        if len(Ts) < 4:
            continue
        th = [x['Theta'] / P for t in Ts for x in byT[t] if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    T0h = float(np.sqrt(max(los) * min(his)))
    print(f'common reduced Theta_0 = {T0h:.3e}\n')

    print('=' * 78)
    print('TWO-LEVER TEST on the anisotropy LEVELS')
    print('=' * 78)
    print('   friction lever at one pressure vs pressure lever, exactly as for chi\n')

    for var in ('a_c', 'a_n', 'a_t'):
        pts = {}
        for P in Ps:
            T0 = T0h * P
            for mg in mus:
                rr = cm.get((P, mg))
                if not rr:
                    continue
                f = nfit.fit_local_sys(rr, T0)
                x = level_at(rows, cm, P, mg, T0, var)
                if f and x is not None:
                    pts.setdefault(P, []).append((x, f['n'], f['tot'], mg))
        if len(pts) < 2:
            print(f'   {var}: insufficient coverage')
            continue
        lo, hi = min(pts), max(pts)
        print(f'   --- {var} ---')
        for P in (lo, hi):
            rng = [p[0] for p in pts[P]]
            print(f'      P={P:g}: {var} spans {min(rng):.4f} to {max(rng):.4f}')
        A = sorted(pts[lo]); B = sorted(pts[hi])
        fx = np.array([p[0] for p in A]); fy = np.array([p[1] for p in A])
        fe = np.array([p[2] for p in A])
        d, errs = [], []
        for c, n, e, mg in B:
            if c < fx.min() or c > fx.max():
                continue
            pred = float(np.interp(c, fx, fy))
            perr = float(np.interp(c, fx, fe))
            d.append(n - pred); errs.append(float(np.hypot(e, perr)))
        if len(d) < 3:
            print(f'      only {len(d)} overlapping points -- no usable test\n')
            continue
        d = np.array(d)
        ref_sys = float(np.mean(fe))
        se = float(np.sqrt(np.mean(errs) ** 2 / len(d) + ref_sys ** 2))
        rms = float(np.sqrt(np.mean((d / np.array(errs)) ** 2)))
        print(f'      {len(d)} overlapping points')
        print(f'      RMS discrepancy   {rms:.1f} sigma')
        print(f'      coherent offset   {d.mean():+.4f} +/- {se:.4f}'
              f'  -> {abs(d.mean())/se:.1f} sigma   ({int((d<0).sum())}/{len(d)} negative)')
        verdict = ('FAILS -- not a state function' if abs(d.mean()) / se >= 2 or rms >= 2
                   else 'no failure detected')
        print(f'      -> {verdict}')
        span = float(fy.max() - fy.min())
        print(f'      power: 2-sigma detectable {2*np.mean(errs):.4f} against an '
              f'n-span of {span:.4f}'
              + ('   [UNDER-POWERED]' if 2 * np.mean(errs) > 0.5 * span else ''))
        print()

    print('   Prediction under the decomposition: all three FAIL, because n depends')
    print('   on the channel DERIVATIVES, not on the anisotropy levels. A collapse')
    print('   here would contradict §5.')


if __name__ == '__main__':
    main()
