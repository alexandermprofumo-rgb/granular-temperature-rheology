"""Constraints 7, 8 and 10 re-founded on strain-1.0 data.

All three rested on strain-0.12 runs, and the two-lever tests were the results
most exposed: a lever compares n at two (E, P, k_t) states, and the convergence
strain depends on stiffness and pressure, so each arm was measured at a
different fraction of its own convergence. Every arm here reaches the same
STRAIN (step counts computed per cell from gdot*dt), all decks dump the
tangential force so chi is measured on the runs that give n, and Z is recorded
so the jamming gate applies.

Errors use nfit's calibrated systematic; slopes use propagated covariance, not
residual-scaled (the section 10.4 bug that levers_local.py still had).

Usage:  python3 analyze_levers_steady.py
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit

ZMIN = 3.0
MOB = 0.99


def cells(pattern, rx, keyf):
    out = {}
    for k, v in nfit.load_sweep(pattern, rx).items():
        out.setdefault(keyf(dict(k)), []).extend(v)
    return out


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if not z or np.mean(z) >= ZMIN:
            keep.append(t)
    return keep


def fit(runs, T0):
    ts = jammed(runs)
    return nfit.fit_local_sys(runs, T0, restrict_to=ts) if len(ts) >= 4 else None


def wslope(x, y, e):
    """Propagated covariance, never rescaled by residuals."""
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(e, float)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float((((y - A @ c) / e) ** 2).sum()) / max(len(x) - 2, 1)
    return float(c[1]), float(np.sqrt(cov[1, 1])), chi2


# ---------------------------------------------------------------- chi
def chi_cache(d, rxd, name):
    if os.path.exists(name):
        rows = list(csv.DictReader(open(name)))
        for r in rows:
            for k in ('mu_g', 'Tgran', 'chi'):
                r[k] = float(r[k])
            for k in ('E', 'kt'):
                if k in r:
                    r[k] = float(r[k])
        return rows
    from analyze_pichi import frames
    rows = []
    paths = sorted(glob.glob(f'{d}/dump.contacts.*'))
    for i, p in enumerate(paths):
        m = rxd.match(os.path.basename(p))
        if not m:
            continue
        g = m.groupdict()
        mg = float(g['mu'])
        vals = []
        for a in frames(p):
            fn = np.linalg.norm(a[:, 6:8], axis=1)
            live = fn > 0
            if live.sum() < 100:
                continue
            vals.append(float((a[live, 10] / (mg * fn[live]) >= MOB).mean()))
        if not vals:
            continue
        row = dict(mu_g=mg, Tgran=float(g['T']), seed=g['s'],
                   chi=float(np.mean(vals)))
        for k in ('E', 'kt'):
            if k in g:
                row[k] = float(g[k])
        rows.append(row)
        if (i + 1) % 60 == 0:
            print(f'   ... chi {i+1}/{len(paths)}', file=sys.stderr)
    with open(name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    RXE = (r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
           r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    RXK = (r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
           r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    RXT = (r'log\.td(?P<td>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
           r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    E = cells('sweep_lev_E/log.E*', RXE, lambda d: (float(d['E']), float(d['mu'])))
    K = cells('sweep_lev_kt/log.k*', RXK, lambda d: (float(d['kt']), float(d['mu'])))
    T = cells('sweep_lev_td/log.td*', RXT, lambda d: (float(d['td']), float(d['mu'])))

    # ------------------------------------------------ constraint 7
    print('=' * 78)
    print('CONSTRAINT 7   stiffness derivatives at strain 1.0')
    print('=' * 78)
    for mg in (0.1, 0.3, 1.0):
        out = {}
        for nm, C, ref in (('dn/dln(E)', E, 1.0e5), ('dn/dln(k_t)', K, 9.05e4)):
            arr = sorted((k[0], v) for k, v in C.items() if abs(k[1] - mg) < 1e-9)
            got = nfit.common_theta0([v for _, v in arr])
            if got is None:
                continue
            T0, _ = got
            xs, ys, es = [], [], []
            for x, v in arr:
                r = fit(v, T0)
                if r:
                    xs.append(np.log(x / ref)); ys.append(r['n']); es.append(r['tot'])
            if len(xs) < 3:
                continue
            s, se, c2 = wslope(xs, ys, es)
            out[nm] = (s, se)
            print(f'   mu_g={mg:<5g} {nm:<12} {s:+.4f} +/- {se:.4f} '
                  f'({abs(s)/se:4.1f}s)  chi2/dof {c2:.2f}  [{len(xs)} pts, '
                  f'Theta_0={T0:.2e}]')
        if len(out) == 2:
            a, ae = out['dn/dln(E)']; b, be = out['dn/dln(k_t)']
            kn, kne = a - b, float(np.hypot(ae, be))
            print(f'   {"":<11} => dn/dln(k_n) {kn:+.4f} +/- {kne:.4f} '
                  f'({abs(kn)/kne:4.1f}s)   opposite signs: {(kn>0)!=(b>0)}'
                  f'   weaker of pair {min(abs(kn)/kne, abs(b)/be):.1f}s')

    # ------------------------------------------------ constraint 8: chi
    print()
    print('=' * 78)
    print('CONSTRAINT 8   chi under the STIFFNESS lever  (the deciding test)')
    print('=' * 78)
    ce = chi_cache('sweep_lev_E',
                   re.compile(r'dump\.contacts\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                              r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'), 'lev_E_chi.csv')
    Es = sorted({k[0] for k in E})
    mus = sorted({k[1] for k in E})
    ref, test = 1.0e5, max(Es)
    pool = [E[(e, m)] for e in (ref, test) for m in mus if (e, m) in E]
    T0, _ = nfit.common_theta0(pool)
    print(f'   reference E = {ref:g}, testing E = {test:g}, Theta_0 = {T0:.3e}')
    P = {}
    for e in (ref, test):
        for m in mus:
            if (e, m) not in E:
                continue
            r = fit(E[(e, m)], T0)
            ts = {round(t, 12) for t in jammed(E[(e, m)])}
            cv = [x['chi'] for x in ce if x.get('E') == e and x['mu_g'] == m
                  and round(x['Tgran'], 12) in ts]
            if r and cv:
                P[(e, m)] = (float(np.mean(cv)), r['n'], r['tot'])
    A = sorted(P[(ref, m)] for m in mus if (ref, m) in P)
    B = [P[(test, m)] + (m,) for m in mus if (test, m) in P]
    fx = np.array([a[0] for a in A]); fy = np.array([a[1] for a in A])
    fe = np.array([a[2] for a in A])
    d, ee = [], []
    for c_, n_, e_, m in sorted(B):
        if c_ < fx.min() or c_ > fx.max():
            continue
        pr = float(np.interp(c_, fx, fy)); pe = float(np.interp(c_, fx, fe))
        d.append(n_ - pr); ee.append(float(np.hypot(e_, pe)))
        print(f'      mu_g={m:<5g} chi={c_:.4f}  n={n_:+.4f}  predicted {pr:+.4f}'
              f'   {(n_-pr)/np.hypot(e_,pe):+.1f} sigma')
    if len(d) >= 3:
        d = np.array(d); ee = np.array(ee)
        mo = float(d.mean()); se = float(np.sqrt((ee ** 2).mean() / len(d)))
        print(f'\n   coherent offset {mo:+.4f} +/- {se:.4f} -> {abs(mo)/se:.1f} sigma'
              f'   ({int((d<0).sum())}/{len(d)} negative)')
        print(f'   RMS {np.sqrt(((d/ee)**2).mean()):.1f} sigma')
        print(f'   [pressure lever gave 0.5 sigma; strain-0.12 pressure gave 3.9]')

    # ------------------------------------------------ constraint 10
    print()
    print('=' * 78)
    print('CONSTRAINT 10   n vs thermostat coupling  (Theta window now ~3x wider)')
    print('=' * 78)
    for mg in (0.1, 0.3, 1.0):
        arr = sorted((k[0], v) for k, v in T.items() if abs(k[1] - mg) < 1e-9)
        got = nfit.common_theta0([v for _, v in arr])
        if got is None:
            continue
        T0, (lo, hi) = got
        xs, ys, es = [], [], []
        for x, v in arr:
            r = fit(v, T0)
            if r:
                xs.append(np.log(x)); ys.append(r['n']); es.append(r['tot'])
        if len(xs) < 3:
            continue
        s, se, c2 = wslope(xs, ys, es)
        span = max(xs) - min(xs)
        print(f'   mu_g={mg:<5g} dn/dln(Pi_damp) = {s:+.4f} +/- {se:.4f} '
              f'({abs(s)/se:.1f}s)  chi2/dof {c2:.2f}')
        print(f'   {"":<11} Theta window {hi/lo:.1f}x  [was 1.9-2.8x]; '
              f'2-sigma bound on variation over the {np.exp(span):.0f}x coupling '
              f'range: {2*se*span:.4f}')


if __name__ == '__main__':
    main()
