"""levers_local.wslope rescales the covariance by the fit residuals.

That must never be done when dof is small:
"A three-point fit reported b = -0.0326 +/- 0.0057 (5.7 sigma) ... With errors
propagated from the data instead of the residuals, both are ~1.0 sigma."

The stiffness derivatives of constraint 7 use 4-5 points, i.e. dof = 2-3, and
are computed with exactly that estimator.  Recompute both ways.
"""
import numpy as np
import nfit
from levers_local import cells, wslope


def wslope_propagated(xs, ns, es):
    """Same fit, covariance = inv(A^T W A): NOT rescaled by residuals."""
    x = np.asarray(xs, float); y = np.asarray(ns, float); e = np.asarray(es, float)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1.0 / e ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float(((y - A @ c) ** 2 / e ** 2).sum())
    dof = max(len(x) - 2, 1)
    return float(c[1]), float(np.sqrt(cov[1, 1])), chi2 / dof


st = cells('sweep_stiffness/log.E*',
           r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
kt = cells('sweep_kt_P10/log.k*',
           r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')

print(f'{"cell":<22}{"as coded (residual-scaled)":>30}{"errors propagated":>26}'
      f'{"chi2/dof":>10}')
print('-' * 88)
res = {}
for mg in (0.1, 0.3, 1.0):
    E = sorted((float(dict(k)['E']), v) for k, v in st.items()
               if abs(float(dict(k)['mu']) - mg) < 1e-9)
    K = sorted((float(dict(k)['kt']), v) for k, v in kt.items()
               if abs(float(dict(k)['mu']) - mg) < 1e-9)
    for nm, arr, ref in (('dn/dln(E)', E, 1e5), ('dn/dln(k_t)', K, 7.326e4)):
        got = nfit.common_theta0([v for _, v in arr])
        if got is None:
            continue
        T0, _ = got
        xs, ns, es = [], [], []
        for x, v in arr:
            r = nfit.fit_local_sys(v, T0)
            if r:
                xs.append(np.log(x / ref)); ns.append(r['n']); es.append(r['tot'])
        if len(xs) < 3:
            continue
        s0, e0 = wslope(xs, ns, es)
        s1, e1, c2 = wslope_propagated(xs, ns, es)
        res[(mg, nm)] = (s1, e1)
        print(f'{"mu="+format(mg,"g")+"  "+nm:<22}'
              f'{s0:>+12.4f} +/- {e0:.4f} ({abs(s0)/e0:4.1f}s)'
              f'{s1:>+12.4f} +/- {e1:.4f} ({abs(s1)/e1:4.1f}s)'
              f'{c2:>10.2f}')
        print(f'{"":<22}   n values: '
              + ', '.join(f'{n:+.3f}+/-{e:.3f}' for n, e in zip(ns, es)))
    if (mg, 'dn/dln(E)') in res and (mg, 'dn/dln(k_t)') in res:
        a, ae = res[(mg, 'dn/dln(E)')]; b, be = res[(mg, 'dn/dln(k_t)')]
        kn, kne = a - b, float(np.hypot(ae, be))
        print(f'{"  => dn/dln(k_n)":<22}{"":>30}{kn:>+12.4f} +/- {kne:.4f} '
              f'({abs(kn)/kne:4.1f}s)')
    print()
