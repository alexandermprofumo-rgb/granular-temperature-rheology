"""Analysis of the steady-state campaign (sweep_steady2d / sweep_steady3d).

Three jobs, in order of importance:

  1. THE SELF-CHECK. Refit n on the first and second half of each measurement
     window. The campaign was sized from a measured convergence strain (0.06)
     with an 8x margin, not from a guess -- but a cell whose halves disagree is
     not converged and must be re-run longer INDIVIDUALLY. This is the check
     that makes the short window safe.

  2. THE CORRECTION. n(new) - n(old) per cell, at a common Theta_0, which is
     the transient bias the old production runs carried.

  3. THE REBUILT CONSTRAINTS, from the new runs alone.

Everything goes through nfit with the calibrated systematic (Tier 2).

Usage:  python3 check_steady_campaign.py [2d|3d|both]
"""
import glob
import os
import re
import sys

import numpy as np

import nfit

RX = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
DT, GDOT = 1e-3, 1e-3


def by_mu(pattern):
    out = {}
    for k, v in nfit.load_sweep(pattern, RX).items():
        out.setdefault(float(dict(k)['mu']), []).extend(v)
    return out


def parse_half(path, half):
    """parse_log restricted to the first or second half of the window."""
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
    if len(rows) < 8:
        return None
    a = np.array(rows)
    m = len(a) // 2
    a = a[:m] if half == 0 else a[m:]
    c = {k: a[:, j] for j, k in enumerate(h)}
    out = dict(mu=float(c['v_muI'].mean()), Theta=float(c['v_Theta'].mean()),
               I=float(c['v_Iiner'].mean()))
    if 'v_P' in c:
        out['Pm'] = float(c['v_P'].mean()); out['Ps'] = float(c['v_P'].std())
    return out


def halves(d):
    out = [{}, {}]
    for p in glob.glob(f'{d}/log.mu*'):
        m = re.match(RX, os.path.basename(p))
        if not m:
            continue
        for hf in (0, 1):
            q = parse_half(p, hf)
            if q:
                q['Tgran'] = float(m.group('T'))
                out[hf].setdefault(float(m.group('mu')), []).append(q)
    return out


def drift(d):
    """mu(late)/mu(early) inside the window -- the raw non-steadiness signal."""
    r = []
    for p in glob.glob(f'{d}/log.mu*'):
        a, b = parse_half(p, 0), parse_half(p, 1)
        if a and b and a['mu'] > 0:
            r.append(b['mu'] / a['mu'] - 1)
    return np.array(r)


def report(tag, new_dir, old_pat):
    new = by_mu(f'{new_dir}/log.mu*')
    old = by_mu(old_pat)
    if not new:
        print(f'{tag}: no runs in {new_dir} yet')
        return
    dr = drift(new_dir)
    print('=' * 78)
    print(f'{tag}   {sum(len(v) for v in new.values())} runs, '
          f'{len(new)} frictions   (strain 0.5 equilibration + 0.5 measurement)')
    print('=' * 78)
    print(f'   in-window mu drift: mean {dr.mean():+.2%}, '
          f'{100*(dr > 0).mean():.0f}% positive     '
          f'[old production: +14.9% (2D) / +23.9% (3D), 97-99% positive]')

    H = halves(new_dir)
    mus = sorted(set(new) & set(old))
    print()
    print(f'{"mu_g":>7}{"n (steady)":>21}{"n (old, strain .12)":>22}'
          f'{"shift":>9}{"half-split":>12}  conv')
    rows = []
    for mg in mus:
        got = nfit.common_theta0([new[mg], old[mg]])
        if got is None:
            print(f'{mg:>7g}   no common Theta range with the old sweep')
            continue
        T0, _ = got
        a = nfit.fit_local_sys(new[mg], T0)
        b = nfit.fit_local_sys(old[mg], T0)
        if not (a and b):
            print(f'{mg:>7g}   unfittable')
            continue
        hs = []
        for hf in (0, 1):
            if mg in H[hf]:
                r = nfit.fit_local(H[hf][mg], T0)
                if r:
                    hs.append(r['n'])
        hd = (hs[1] - hs[0]) if len(hs) == 2 else np.nan
        conv = 'ok' if (np.isfinite(hd) and abs(hd) < 2 * a['tot']) else \
               ('CHECK' if np.isfinite(hd) else '-')
        d = a['n'] - b['n']
        rows.append((mg, a, b, d, hd))
        print(f'{mg:>7g}{a["n"]:>+13.4f}+/-{a["tot"]:<7.4f}'
              f'{b["n"]:>+14.4f}+/-{b["tot"]:<7.4f}{d:>+9.4f}'
              f'{hd:>+12.4f}  {conv}')
    if rows:
        dd = np.array([r[3] for r in rows])
        print(f'\n   transient bias removed: mean {dd.mean():+.4f}, '
              f'range {dd.min():+.4f} to {dd.max():+.4f}')
        lo = [r for r in rows if r[0] <= 0.1]
        hi = [r for r in rows if r[0] > 0.1]
        if lo:
            print(f'      at mu_g <= 0.1 : {np.mean([r[3] for r in lo]):+.4f}'
                  f'   (predicted ~0)')
        if hi:
            print(f'      at mu_g >  0.1 : {np.mean([r[3] for r in hi]):+.4f}'
                  f'   (predicted ~-0.03)')

    # -- constraints rebuilt from the new runs alone --
    core = [m for m in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0) if m in new]
    if len(core) >= 5:
        got = nfit.common_theta0([new[m] for m in core])
        if got:
            T0, _ = got
            R = {m: nfit.fit_local_sys(new[m], T0) for m in core}
            R = {m: v for m, v in R.items() if v}
            print(f'\n   --- rebuilt from steady runs, Theta_0 = {T0:.3e} ---')
            if 0.0 in R:
                pred = 0.25 if tag.startswith('2D') else 1.0 / 6.0
                r = R[0.0]
                print(f'   C2 frictionless: {r["n"]:.4f} +/- {r["tot"]:.4f} '
                      f'vs {pred:.4f} -> {abs(r["n"]-pred)/r["tot"]:.1f} sigma')
            if len(R) >= 4 and 0.0 in R:
                pk = max(R, key=lambda m: R[m]['n'])
                e = float(np.hypot(R[pk]['tot'], R[0.0]['tot']))
                print(f'   C3/4 peak {R[pk]["n"]:.4f} at mu_g = {pk:g};  '
                      f'rise {(R[pk]["n"]-R[0.0]["n"])/e:.1f} sigma')
                if 1.0 in R:
                    e2 = float(np.hypot(R[pk]['tot'], R[1.0]['tot']))
                    print(f'        fall to mu_g = 1: '
                          f'{(R[pk]["n"]-R[1.0]["n"])/e2:.1f} sigma')
    print()


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'both').lower()
    if which in ('2d', 'both'):
        report('2D', 'sweep_steady2d', 'sweep_matched2d/log.mu*')
    if which in ('3d', 'both'):
        report('3D', 'sweep_steady3d', 'sweep_run_3d/log.mu*')


if __name__ == '__main__':
    main()
