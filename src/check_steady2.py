"""Production (strain 0.12) vs steady state (strain 1.5), done fairly.

sweep_steadystate has only 4 setpoints, and fit_local_sys silently returns
sys = NaN there (the jackknife refits drop to 3 setpoints, fit_local needs 4,
so vals is empty and tot collapses to stat with no warning).  So compare
like with like: restrict BOTH sides to the 4 shared setpoints, and quote both
estimators.
"""
import numpy as np
import nfit


def by_mu(pat, rx):
    out = {}
    for k, v in nfit.load_sweep(pat, rx).items():
        out.setdefault(float(dict(k)['mu']), []).extend(v)
    return out


RX = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
ss = by_mu('sweep_steadystate/log.mu*', RX)
m2 = by_mu('sweep_matched2d/log.mu*', RX)

print('production: nequil 40000 / nmeas 80000 -> total strain 0.12')
print('steadystate: nequil 500000 / nmeas 1000000 -> total strain 1.5\n')
for mg in sorted(ss):
    ta = {round(r['Tgran'], 12) for r in m2[mg]}
    tb = {round(r['Tgran'], 12) for r in ss[mg]}
    win = sorted(ta & tb)
    print(f'mu_g = {mg}   shared setpoints: {[f"{t:g}" for t in win]}')

    # (i) window-averaged n on the shared window -- what replica_audit does
    a = nfit.fit_n(m2[mg], restrict_to=win)
    b = nfit.fit_n(ss[mg], restrict_to=win)
    d = b['n'] - a['n']
    for tag, ea, eb in (('stat', a['stat'], b['stat']),
                        ('tot ', a['tot'], b['tot'])):
        e = float(np.hypot(ea, eb))
        print(f'   fit_n  ({tag}): production {a["n"]:+.4f}+/-{ea:.4f}  '
              f'steady {b["n"]:+.4f}+/-{eb:.4f}   diff {d:+.4f}  {abs(d)/e:.1f} sigma')

    # (ii) the estimator actually quoted, on the shared window
    got = nfit.common_theta0([m2[mg], ss[mg]], restrict_to=win)
    if got:
        T0, _ = got
        A = nfit.fit_local_sys(m2[mg], T0, restrict_to=win)
        B = nfit.fit_local_sys(ss[mg], T0, restrict_to=win)
        if A and B:
            d = B['n'] - A['n']
            e = float(np.hypot(A['stat'], B['stat']))
            print(f'   fit_local (stat, Theta_0={T0:.2e}): production '
                  f'{A["n"]:+.4f}+/-{A["stat"]:.4f}  steady {B["n"]:+.4f}'
                  f'+/-{B["stat"]:.4f}   diff {d:+.4f}  {abs(d)/e:.1f} sigma')
            print(f'      (sys returned by fit_local_sys: production '
                  f'{A["sys"]}, steady {B["sys"]}  <- NaN means the jackknife '
                  f'silently vanished)')

    # (iii) and the level of mu itself
    for tag, C in (('production', m2[mg]), ('steady    ', ss[mg])):
        Ts, byT = nfit.surviving_setpoints(C, restrict_to=win)
        mus = [np.mean([r['mu'] for r in byT[t]]) for t in Ts]
        print(f'   mu at shared setpoints, {tag}: '
              + ', '.join(f'{m:.4f}' for m in mus))
    print()
