"""Calibrate nfit's error budget on REAL cell geometry.

For each real cell: fit the quadratic, then generate 400 parametric replicas
that use the SAME Theta values and the SAME residual scale but are drawn from
the fitted quadratic exactly -- so there is, by construction, zero window
systematic.  Compare:
    truth = scatter of n over replicas
    stat  = what fit_local reports
    sys   = what fit_local_sys reports as the "setpoint jackknife systematic"
    tot   = hypot(stat, sys), the number every constraint in section 8 uses.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def load(pat):
    raw = nfit.load_sweep(pat, r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    C = {}
    for key, runs in raw.items():
        C.setdefault(float(dict(key)['mu']), []).extend(runs)
    return C


m2 = load('sweep_matched2d/log.mu*')
d3 = load('sweep_run_3d/log.mu*')
pool = [m2[m] for m in CORE if m in m2] + [d3[m] for m in CORE if m in d3]
T0, _ = nfit.common_theta0(pool)

rng = np.random.default_rng(0)
print(f'Theta_0 = {T0:.3e}\n')
print(f'{"cell":<12}{"stat":>9}{"sys":>9}{"tot":>9}{"TRUTH":>9}'
      f'{"tot/truth":>11}{"stat/truth":>12}')
print('-' * 71)
summary = []
for tag, C in (('2D', m2), ('3D', d3)):
    for mg in CORE:
        if mg not in C:
            continue
        runs = C[mg]
        r = nfit.fit_local_sys(runs, T0)
        if not r or not np.isfinite(r['sys']):
            continue
        Ts, byT = nfit.surviving_setpoints(runs)
        sel = [x for t in Ts for x in byT[t]]
        th = np.array([x['Theta'] for x in sel])
        mu = np.array([x['mu'] for x in sel])
        u = np.log(th / T0)
        A = np.vstack([np.ones_like(u), u, u ** 2]).T
        c, *_ = np.linalg.lstsq(A, np.log(mu), rcond=None)
        resid = np.log(mu) - A @ c
        s = resid.std(ddof=3)
        ns = []
        for _ in range(400):
            fake = [dict(x, mu=float(np.exp((A @ c)[i] + rng.normal(0, s))))
                    for i, x in enumerate(sel)]
            f = nfit.fit_local(fake, T0)
            if f:
                ns.append(f['n'])
        truth = float(np.std(ns))
        summary.append((r['stat'], r['sys'], r['tot'], truth))
        print(f'{tag+" mu="+format(mg,"g"):<12}{r["stat"]:>9.4f}{r["sys"]:>9.4f}'
              f'{r["tot"]:>9.4f}{truth:>9.4f}{r["tot"]/truth:>11.2f}'
              f'{r["stat"]/truth:>12.2f}')

S = np.array(summary)
print('-' * 71)
print(f'{"MEAN":<12}{S[:,0].mean():>9.4f}{S[:,1].mean():>9.4f}{S[:,2].mean():>9.4f}'
      f'{S[:,3].mean():>9.4f}{(S[:,2]/S[:,3]).mean():>11.2f}'
      f'{(S[:,0]/S[:,3]).mean():>12.2f}')
