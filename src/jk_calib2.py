"""What does nfit's setpoint jackknife return when there is, by construction,
NOTHING for it to find?

Same real cells, same Theta values, same residual scale -- but mu regenerated
from the fitted quadratic exactly.  Any 'sys' returned is pure estimator noise.
Compare that null expectation with the sys actually quoted for the real data.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
NREP = 200


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

print(f'Theta_0 = {T0:.3e}    ({NREP} null replicas per cell)\n')
print(f'{"cell":<12}{"k":>3}{"stat":>8}{"sys(real)":>11}{"sys(null)":>11}'
      f'{"excess":>9}{"sys/null":>10}   verdict')
print('-' * 78)
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
        u = np.log(np.array([x['Theta'] for x in sel]) / T0)
        y = np.log(np.array([x['mu'] for x in sel]))
        A = np.vstack([np.ones_like(u), u, u ** 2]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        s = (y - A @ c).std(ddof=3)
        nulls = []
        for _ in range(NREP):
            fake = [dict(x, mu=float(np.exp((A @ c)[i] + rng.normal(0, s))))
                    for i, x in enumerate(sel)]
            f = nfit.fit_local_sys(fake, T0)
            if f and np.isfinite(f['sys']):
                nulls.append(f['sys'])
        nl = float(np.mean(nulls))
        exc = float(np.sqrt(max(r['sys'] ** 2 - nl ** 2, 0.0)))
        v = 'no evidence of misfit' if r['sys'] < nl * 1.2 else \
            ('real window sensitivity' if r['sys'] > nl * 1.8 else 'marginal')
        print(f'{tag+" mu="+format(mg,"g"):<12}{r["k"]:>3}{r["stat"]:>8.4f}'
              f'{r["sys"]:>11.4f}{nl:>11.4f}{exc:>9.4f}{r["sys"]/nl:>10.2f}   {v}')
