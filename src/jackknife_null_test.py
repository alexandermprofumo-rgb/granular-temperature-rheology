"""Does nfit's 'systematic' jackknife measure a systematic, or re-measure the
statistical error?

Synthetic data drawn from an EXACTLY quadratic ln mu(ln Theta) with iid
Gaussian noise: there is no window systematic at all, by construction.  If
fit_local_sys still returns sys ~ stat, then tot = hypot(stat, sys) inflates
every error bar by ~sqrt(2) for no reason, and 'sys/stat ~ 1' in the real
tables is the null expectation rather than evidence of model inadequacy.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

rng = np.random.default_rng(7)

# geometry copied from a real 2D cell: 8 setpoints, 4 seeds, Theta 3e-4..1.5e-3
TS = np.geomspace(3.0e-4, 1.55e-3, 8)
NSEED = 4
THETA0 = float(np.sqrt(TS[0] * TS[-1]))
NTRUE, CURV, SIG = 0.20, 0.05, 0.004     # sigma on ln mu, ~ real residual size


def make(noise=SIG, curv=CURV):
    runs = []
    for t in TS:
        for _ in range(NSEED):
            th = t * np.exp(rng.normal(0, 0.01))
            u = np.log(th / THETA0)
            lnmu = -NTRUE * u - 0.5 * curv * u ** 2 + rng.normal(0, noise)
            runs.append(dict(Tgran=t, Theta=th, mu=float(np.exp(lnmu)),
                             I=1.0, Pm=10.0, Ps=0.01))
    return runs


for label, curv in (('exact quadratic (no misfit possible)', CURV),
                    ('pure power law (no curvature)', 0.0)):
    S, J, N = [], [], []
    for _ in range(400):
        r = nfit.fit_local_sys(make(curv=curv), THETA0)
        if r and np.isfinite(r['sys']):
            S.append(r['stat']); J.append(r['sys']); N.append(r['n'])
    S, J, N = map(np.array, (S, J, N))
    print(f'\n{label}')
    print(f'   true n = {NTRUE}')
    print(f'   actual scatter of fitted n over 400 realisations : {N.std():.5f}')
    print(f'   mean quoted stat                                 : {S.mean():.5f}')
    print(f'   mean quoted "sys" (setpoint jackknife)           : {J.mean():.5f}')
    print(f'   mean sys/stat                                    : {(J/S).mean():.2f}')
    print(f'   mean quoted total = hypot(stat,sys)              : '
          f'{np.hypot(S, J).mean():.5f}')
    print(f'   -> total / true scatter                          : '
          f'{np.hypot(S, J).mean()/N.std():.2f}x')
