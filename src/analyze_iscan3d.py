"""The 3D inertial-number scan: is the FORM failure dimension-dependent?

WHY THIS EXISTS. The paper's title claim is that mu*Theta^n = F(I) is not
always valid, and its strongest leg is the FORM leg -- Theta and I are not
separable, so no value of n rescues the relation. As of the 2026-08-25 audit
that leg was 2D ONLY: every 3D run in the project sat at a single shear rate,
I = 3.24e-4, which leaves the dimension dependence of the form failure
untested.

sweep_iscan3d supplies the two outer arms, sharing sweep_steady3d's packings so
that the existing campaign IS the middle arm and all three are paired:

    slow   gdot = 3.162e-4   I = 1.0e-4    sweep_iscan3d
    mid    gdot = 1.0e-3     I = 3.16e-4   sweep_steady3d
    fast   gdot = 3.162e-3   I = 1.0e-3    sweep_iscan3d

THE TEST IS ANALYSIS_PROTOCOL Amendment 3: the pre-registered nested F-test
(shared shape against free per-arm quadratics) and the Theta_0-INDEPENDENT
beta = gamma = 0 test, both on all points and on setpoint means. Separability
is a statement about the whole window, not about one Theta_0, and the 2D
presentation that quoted dn/dlnI at three positions in the window is not
repeated here.

Prediction on record (Amendment 5b, before the data existed): fails at
mu_g = 0.3, holds at mu_g = 0, as in 2D.

Usage:  python3 analyze_iscan3d.py
"""
import numpy as np
from scipy import stats

import nfit

ZMIN = 4.0                      # D+1 in 3D
MUS = [0.0, 0.15, 0.3]
RX_G = (r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
        r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
RX_M = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
ARMS = [('slow', 1.000e-4, 'sweep_iscan3d/log.g3.162e-4_mu*', RX_G),
        ('mid',  3.162e-4, 'sweep_steady3d/log.mu*',          RX_M),
        ('fast', 1.000e-3, 'sweep_iscan3d/log.g3.162e-3_mu*', RX_G)]
I_MID = 3.162e-4


def load(pattern, rx):
    out = {}
    for key, runs in nfit.load_sweep(pattern, rx).items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


def gated(runs, window=None):
    """Jamming-gated setpoints, optionally restricted to a shared Theta window."""
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if z and np.mean(z) < ZMIN:
            continue
        if window is not None:
            th = [r['Theta'] for r in byT[t] if r['Theta'] > 0]
            if not th or not (window[0] * 0.999 <= np.mean(th) <= window[1] * 1.001):
                continue
        keep.append(t)
    return keep, byT


def _rss(A, y):
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ c
    return float(r @ r)


def main():
    data = {tag: (I, load(p, rx)) for tag, I, p, rx in ARMS}
    print('=' * 78)
    print('THE 3D I-SCAN   --   is the separability failure dimension-dependent?')
    print('=' * 78)
    for tag, I, _, _ in ARMS:
        print(f'  {tag:<5} I = {I:.3e}   {len(data[tag][1])} frictions present')
    print()

    for mg in MUS:
        pools = [(tag, data[tag][0], data[tag][1][mg])
                 for tag, _, _, _ in ARMS if mg in data[tag][1]]
        if len(pools) < 3:
            print(f'  mu_g = {mg}: only {len(pools)} arms present -- skipped\n')
            continue
        got = nfit.common_theta0([p[2] for p in pools])
        if got is None:
            print(f'  mu_g = {mg}: no common Theta range\n')
            continue
        T0, (lo, hi) = got
        print(f'=== mu_g = {mg}   shared window [{lo:.3e}, {hi:.3e}] = {hi/lo:.1f}x'
              f'   Theta_0 = {T0:.3e}')

        rec, arms, percell = [], {}, []
        for tag, I, runs in pools:
            ts, byT = gated(runs, window=(lo, hi))
            f = nfit.fit_local_sys(runs, T0, restrict_to=ts, z_min=ZMIN) if len(ts) >= 4 else None
            pts = sorted((float(np.mean([r['Theta'] for r in byT[t]])),
                          float(np.mean([np.log(r['mu']) for r in byT[t]])))
                         for t in ts)
            arms[tag] = pts
            if f:
                percell.append((f['n'], f['tot']))
                print(f'   {tag:<5} I={I:.3e}  k={f["k"]:<3d} '
                      f'Th[{pts[0][0]:.3e},{pts[-1][0]:.3e}]  '
                      f'n = {f["n"]:+.4f} +/- {f["tot"]:.4f}')
            else:
                print(f'   {tag:<5} VOID ({len(ts)} jammed setpoints in window)')
            for t in ts:
                for r in byT[t]:
                    if r['Theta'] > 0 and r['mu'] > 0:
                        rec.append((tag, I, np.log(r['Theta'] / T0),
                                    np.log(r['mu']), False))
                th = [r['Theta'] for r in byT[t] if r['Theta'] > 0 and r['mu'] > 0]
                lm = [np.log(r['mu']) for r in byT[t]
                      if r['Theta'] > 0 and r['mu'] > 0]
                if th:
                    rec.append((tag, I,
                                float(np.mean(np.log(np.array(th) / T0))),
                                float(np.mean(lm)), True))
        if len({r[0] for r in rec}) < 3:
            print('   fewer than three arms survive the gate -- no form test\n')
            continue

        res = {}
        for lvl in (False, True):
            s = [r for r in rec if r[4] is lvl]
            u = np.array([r[2] for r in s]); y = np.array([r[3] for r in s])
            L = np.array([np.log(r[1] / I_MID) for r in s])
            arm = np.array([r[0] for r in s]); one = np.ones_like(u)
            n = len(y)
            Af = np.column_stack([one, L, u, L * u, u ** 2, L * u ** 2])
            Ar = np.column_stack([one, L, u, u ** 2])
            F2 = ((_rss(Ar, y) - _rss(Af, y)) / 2) / (_rss(Af, y) / (n - 6))
            res[lvl] = (F2, 1 - stats.f.cdf(F2, 2, n - 6), n)
            if not lvl:
                d = [(arm == t).astype(float) for t in ('slow', 'mid', 'fast')]
                Asep = np.column_stack(d + [u, u ** 2])
                Afree = np.column_stack([a * b for a in d
                                         for b in [one, u, u ** 2]])
                r1, r2 = _rss(Asep, y), _rss(Afree, y)
                p1, p2 = Asep.shape[1], Afree.shape[1]
                Fn = ((r1 - r2) / (p2 - p1)) / (r2 / (n - p2))
                nested = (Fn, 1 - stats.f.cdf(Fn, p2 - p1, n - p2),
                          np.sqrt(r1 / max(r2, 1e-300)))
        # T1: is n CONSTANT across the arms? Same statistic as analyze_iscan3's
        # T1 block, which had no 3D counterpart. Sec. V.B quotes both
        # dimensions, and at mu_g = 0 they disagree, so the 3D value has to be
        # computed here rather than by hand.
        if len(percell) >= 3:
            v = np.array([c[0] for c in percell])
            e = np.array([c[1] for c in percell])
            w = 1.0 / e ** 2
            mean = float((w * v).sum() / w.sum())
            chi2 = float((w * (v - mean) ** 2).sum())
            print(f'   T1 constant across arms: mean {mean:+.4f}'
                  f' +/- {1/np.sqrt(w.sum()):.4f}   chi2 = {chi2:.2f} on'
                  f' {len(v)-1} dof   '
                  f'{"CONSTANT" if chi2 < 6 else "NOT CONSTANT"}')
        print(f'   nested F (shared shape vs free per-arm quadratics): '
              f'F = {nested[0]:.2f}, p = {nested[1]:.1e}, '
              f'RMS penalty {nested[2]:.2f}x')
        print(f'   beta = gamma = 0 (Theta_0-free): F = {res[False][0]:.2f}, '
              f'p = {res[False][1]:.1e}  [N={res[False][2]}]')
        print(f'   same on setpoint means (cluster-robust): F = {res[True][0]:.2f}, '
              f'p = {res[True][1]:.1e}  [N={res[True][2]}]')
        print(f'   -> {"NOT SEPARABLE" if res[False][1] < 0.05 else "consistent with separable"}')

        # model-free: two-point log-slope per arm, and within sub-windows.
        # No fitting, so no window/degree argument is available against it.
        print(f'   {"model-free two-point slope":<26}'
              + ''.join(f'{t:>10}' for t in ('slow', 'mid', 'fast'))
              + f'{"slow-fast":>12}')
        for lab, sl in (('lower half', lambda q: q[:max(2, len(q) // 2 + 1)]),
                        ('upper half', lambda q: q[len(q) // 2:]),
                        ('interior', lambda q: q[1:-1]),
                        ('full window', lambda q: q)):
            v = {}
            for t in ('slow', 'mid', 'fast'):
                q = sl(arms.get(t, []))
                v[t] = (-(q[-1][1] - q[0][1]) / np.log(q[-1][0] / q[0][0])
                        if len(q) >= 2 else np.nan)
            print(f'   {lab:<26}'
                  + ''.join(f'{v[t]:>10.4f}' for t in ('slow', 'mid', 'fast'))
                  + f'{v["slow"]-v["fast"]:>12.4f}')
        print()

    # The 2D comparison is NOT reproduced here. It was, as three hardcoded
    # F and p values, and they drifted out of date against the script that
    # computes them. Run analyze_iscan3.py for the 2D column.
    print('For the 2D column, run analyze_iscan3.py.')


if __name__ == '__main__':
    main()
