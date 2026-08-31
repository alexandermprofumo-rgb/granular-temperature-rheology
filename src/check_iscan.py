"""Reproduce section 11 (n depends on I) from sweep_iscan.

There is no script in RESEARCH_LAMMPS for this, so it is rebuilt here from the
sweep layout in run_iscan.sh:  gdot in {3.162e-4, 1.0e-3, 3.162e-3}, with the
middle arm taken from sweep_matched2d (which run_iscan.sh says is 'already
run').  I = gdot * d * sqrt(rho/P) = gdot * 1 * sqrt(1/10).
"""
import numpy as np
import nfit

GD = {'3.162e-4': 1.0e-4, '1.0e-3': 3.162e-4, '3.162e-3': 1.0e-3}
MUS = [0.0, 0.15, 0.3, 1.0]


def load_iscan():
    raw = nfit.load_sweep(
        'sweep_iscan/log.g*',
        r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    out = {}
    for key, runs in raw.items():
        d = dict(key)
        out.setdefault((d['g'], float(d['mu'])), []).extend(runs)
    return out


def load_matched():
    raw = nfit.load_sweep('sweep_matched2d/log.mu*',
                          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    out = {}
    for key, runs in raw.items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


isc, m2 = load_iscan(), load_matched()
print('cells found in sweep_iscan:', sorted(isc))
print()

rows = []
for mg in MUS:
    legs = {}
    for g, I in GD.items():
        r = isc.get((g, mg)) if g != '1.0e-3' else m2.get(mg)
        if r:
            legs[I] = r
    if len(legs) < 3:
        print(f'mu_g={mg}: only {len(legs)} arms'); continue
    got = nfit.common_theta0(list(legs.values()))
    if got is None:
        print(f'mu_g={mg}: NO COMMON THETA RANGE'); continue
    T0, (lo, hi) = got
    ns = {}
    for I, r in sorted(legs.items()):
        f = nfit.fit_local_sys(r, T0)
        if f:
            ns[I] = f
    if len(ns) < 3:
        print(f'mu_g={mg}: only {len(ns)} fits'); continue
    x = np.log(np.array(sorted(ns)))
    y = np.array([ns[i]['n'] for i in sorted(ns)])
    e = np.array([ns[i]['tot'] for i in sorted(ns)])
    W = np.vstack([x, np.ones_like(x)]).T
    Ci = np.diag(1 / e ** 2)
    cov = np.linalg.inv(W.T @ Ci @ W)
    c = cov @ W.T @ Ci @ y
    sl, se = float(c[0]), float(np.sqrt(cov[0, 0]))
    rows.append((mg, sl, se))
    print(f'mu_g={mg:<5g} Theta_0={T0:.3e}  common range [{lo:.2e},{hi:.2e}] '
          f'= {hi/lo:.1f}x')
    for I in sorted(ns):
        f = ns[I]
        print(f'      I={I:.2e}  n={f["n"]:+.4f} +/- {f["stat"]:.4f}(stat) '
              f'+/- {f["sys"]:.4f}(sys) = {f["tot"]:.4f}   [{f["k"]} setpoints]')
    print(f'      dn/dlnI = {sl:+.4f} +/- {se:.4f}   ({abs(sl)/se:.1f} sigma)\n')

if rows:
    s = np.array([r[1] for r in rows]); ee = np.array([r[2] for r in rows])
    w = 1 / ee ** 2
    m = float((s * w).sum() / w.sum()); me = float(np.sqrt(1 / w.sum()))
    print(f'weighted mean dn/dlnI = {m:+.4f} +/- {me:.4f}  ({abs(m)/me:.1f} sigma)'
          f'   -> {m*np.log(10):+.4f} per decade')
    chi2 = float((w * (s - m) ** 2).sum())
    print(f'   chi2/dof for a common slope = {chi2/(len(s)-1):.2f}'
          f'   (the four frictions are NOT consistent with one slope if >>1)')
