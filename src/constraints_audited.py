"""
The constraint list with the window systematic restored to n(Theta_0).

constraints_local.py quotes n(Theta_0) +/- stat. That was justified by the
claim that a LOCAL slope has no window ambiguity, unlike a window-averaged n.
The claim is conditional on the fitted model being adequate over the window,
and a direct test shows it is not: a cubic term is required at p < 0.003 in 3D
at every mu_g >= 0.2, and across the 2D crossing band mu_g = 0.06-0.13. Where
the quadratic is inadequate, n(Theta_0) still moves with the window -- by 0.07
to 0.10 in 3D at mu_g = 0.2 and 0.3, three to five times the quoted stat.

So every number here carries the setpoint jackknife alongside stat, and every
significance is computed from the total. Constraints that survive this are
ones a referee cannot dismantle with a window argument; constraints that do
not survive are reported as not surviving.

Usage:  python3 constraints_audited.py
"""
import numpy as np

import nfit

CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def load(pat):
    raw = nfit.load_sweep(pat, r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    C = {}
    for key, runs in raw.items():
        C.setdefault(float(dict(key)['mu']), []).extend(runs)
    return C


def sig(d, e):
    return abs(d) / e if e > 0 else float('nan')


def main():
    m2, d3 = load('sweep_matched2d/log.mu*'), load('sweep_run_3d/log.mu*')
    pool = [m2[m] for m in CORE if m in m2] + [d3[m] for m in CORE if m in d3]
    T0, (lo, hi) = nfit.common_theta0(pool)
    print(f'Theta_0 = {T0:.3e}   (common range [{lo:.2e}, {hi:.2e}])\n')

    R = {}
    for tag, C in (('2D', m2), ('3D', d3)):
        print(f'=== {tag} ===')
        print(f'{"mu_g":<7}{"n":>9}{"stat":>9}{"jk_raw":>9}{"jk_null":>9}'
              f'{"sys":>9}{"total":>9}')
        R[tag] = {}
        for m in CORE:
            if m not in C:
                continue
            r = nfit.fit_local_sys(C[m], T0)
            if not r:
                continue
            if not r['sys_ok']:
                print(f'{m:<7g}   no systematic could be formed -- NOT QUOTABLE')
                continue
            R[tag][m] = r
            print(f'{m:<7g}{r["n"]:>+9.4f}{r["stat"]:>9.4f}{r["sys_raw"]:>9.4f}'
                  f'{r["sys_null"]:>9.4f}{r["sys"]:>9.4f}{r["tot"]:>9.4f}')
        print('   sys = excess of the setpoint jackknife over its own null'
              ' expectation;\n   sys = 0 means the quadratic shows no window'
              ' sensitivity beyond noise.')
        print()

    print('=' * 74)
    print('CONSTRAINT 1/2 -- frictionless baseline and the geometric theory')
    print('=' * 74)
    a, b = R['2D'].get(0.0), R['3D'].get(0.0)
    if a and b:
        d = b['n'] - a['n']
        req = 1.0 / 6.0 - 0.25              # n_3D - n_2D under n = 1/(2D)
        for nm, e in (('stat only', float(np.hypot(a['stat'], b['stat']))),
                      ('stat+sys ', float(np.hypot(a['tot'], b['tot'])))):
            print(f'   {nm}: n_3D - n_2D = {d:+.4f} +/- {e:.4f}  '
                  f'-> dimension-independence {sig(d, e):.1f} sigma; '
                  f'geometric ({req:+.3f}) differs at {sig(d - req, e):.1f} sigma')
        print('   NOTE: the difference is a weak test -- it subtracts two small')
        print('   numbers, and the geometric prediction for it is itself small.')
        print('   The magnitudes are the real test:')
        for nm, r, pred in (('2D', a, 0.25), ('3D', b, 1.0 / 6.0)):
            print(f'      {nm} n = {r["n"]:.4f} +/- {r["tot"]:.4f} (stat+sys) vs '
                  f'predicted {pred:.4f} -> {sig(r["n"] - pred, r["tot"]):.1f} sigma')

    print()
    print('=' * 74)
    print('CONSTRAINT 3/4 -- the peaks')
    print('=' * 74)
    for tag in ('2D', '3D'):
        rows = sorted(R[tag].items())
        if len(rows) < 3:
            continue
        pk = max(rows, key=lambda t: t[1]['n'])
        base, top = R[tag].get(0.0), R[tag].get(1.0)
        print(f'   {tag}: peak n = {pk[1]["n"]:.4f} +/- {pk[1]["tot"]:.4f} '
              f'at mu_g = {pk[0]:g}')
        if base:
            d = pk[1]['n'] - base['n']
            print(f'       rise above frictionless: {d:+.4f} +/- '
                  f'{np.hypot(pk[1]["tot"], base["tot"]):.4f}  -> '
                  f'{sig(d, float(np.hypot(pk[1]["tot"], base["tot"]))):.1f} sigma')
        if top:
            d = pk[1]['n'] - top['n']
            print(f'       fall to mu_g = 1:        {d:+.4f} +/- '
                  f'{np.hypot(pk[1]["tot"], top["tot"]):.4f}  -> '
                  f'{sig(d, float(np.hypot(pk[1]["tot"], top["tot"]))):.1f} sigma')
        # which frictions are statistically indistinguishable from the peak?
        tied = [m for m, r in rows
                if sig(pk[1]['n'] - r['n'], float(np.hypot(pk[1]['tot'], r['tot']))) < 2]
        print(f'       indistinguishable from the peak (<2 sigma): '
              f'{", ".join(f"{t:g}" for t in tied)}')

    print()
    print('=' * 74)
    print('CONSTRAINT 5 -- A_3/A_2')
    print('=' * 74)
    try:
        p2 = max(R['2D'].items(), key=lambda t: t[1]['n'])
        p3 = max(R['3D'].items(), key=lambda t: t[1]['n'])
        e2 = p2[1]['n'] - R['2D'][0.0]['n']
        e3 = p3[1]['n'] - R['3D'][0.0]['n']
        s2 = float(np.hypot(p2[1]['tot'], R['2D'][0.0]['tot']))
        s3 = float(np.hypot(p3[1]['tot'], R['3D'][0.0]['tot']))
        Rr = e3 / e2
        Rs = abs(Rr) * float(np.hypot(s3 / e3, s2 / e2))
        print(f'   enhancement 2D {e2:+.4f} +/- {s2:.4f}, 3D {e3:+.4f} +/- {s3:.4f}')
        print(f'   A_3/A_2 = {Rr:.2f} +/- {Rs:.2f}')
        for t, nm in ((1.0, 'no dimension effect'), (1.5, 'geometric ratio')):
            print(f'      vs {t} ({nm}): {sig(Rr - t, Rs):.1f} sigma')
    except (KeyError, ZeroDivisionError):
        print('   unavailable')


if __name__ == '__main__':
    main()
