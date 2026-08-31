import glob
import numpy as np
import nfit

# ---------------------------------------------------------------- (a) mu(t)
print('=' * 78)
print('(a) mu(t) through equilibration + measurement, one representative run')
print('=' * 78)


def full_trace(path):
    L = open(path).readlines()
    h, rows = None, []
    for l in L:
        t = l.split()
        if t and t[0] == 'Step':
            h = t; continue
        if h is None:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            continue
        if len(v) == len(h):
            rows.append(v)
    a = np.array(rows)
    return {k: a[:, j] for j, k in enumerate(h)}


for lab, p in (('2D matched2d mu=0.3 T=0.003 s1',
                'sweep_matched2d/log.mu0.3_T0.003_s1'),
               ('2D steadystate mu=0.3 T=0.003 s1',
                'sweep_steadystate/log.mu0.3_T0.003_s1')):
    c = full_trace(p)
    st, mu = c['Step'], c['v_muI']
    g = st * 1e-3 * 1e-3
    print(f'\n   {lab}   (final strain {g[-1]:.2f})')
    idx = np.linspace(0, len(st) - 1, 11).astype(int)
    print('      strain: ' + ' '.join(f'{g[i]:7.3f}' for i in idx))
    print('      mu    : ' + ' '.join(f'{mu[i]:7.4f}' for i in idx))

# ------------------------------------------------- (b) the 17/21 sign test
print()
print('=' * 78)
print('(b) section 4: "17 of 21 cells give n(high Theta) > n(low Theta), '
      'p = 0.0072"')
print('=' * 78)


def by_mu(pat, rx):
    out = {}
    for k, v in nfit.load_sweep(pat, rx).items():
        out.setdefault(float(dict(k)['mu']), []).extend(v)
    return out


RX = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
m2 = by_mu('sweep_matched2d/log.mu*', RX)
d3 = by_mu('sweep_run_3d/log.mu*', RX)
up = tot = 0
rows = []
for tag, C in (('2D', m2), ('3D', d3)):
    for mg in sorted(C):
        Ts, byT = nfit.surviving_setpoints(C[mg])
        if len(Ts) < 6:
            continue
        h = len(Ts) // 2
        lo = nfit.fit_n(C[mg], restrict_to=Ts[:h + len(Ts) % 2])
        hi = nfit.fit_n(C[mg], restrict_to=Ts[h:])
        if not (lo and hi):
            continue
        tot += 1
        d = hi['n'] - lo['n']
        up += d > 0
        rows.append((tag, mg, lo['n'], hi['n'], d))
from math import comb
p = 2 * sum(comb(tot, k) for k in range(up, tot + 1)) / 2 ** tot
print(f'   {up} of {tot} cells have n(high Theta) > n(low Theta)   '
      f'two-sided sign test p = {p:.4f}')
print(f'   mean shift {np.mean([r[4] for r in rows]):+.4f}, '
      f'max {max(r[4] for r in rows):+.4f}')
print('   cells: ' + ', '.join(f'{t}{m:g}:{d:+.3f}' for t, m, _, _, d in rows))

# ---------------------------------------------------- (c) gate sensitivity
print()
print('=' * 78)
print('(c) are the gates principled or tuned? n(mu_g=0) and the peak vs I_TOL')
print('=' * 78)
CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
for itol in (0.02, 0.03, 0.05, 0.10, 1.0):
    pool = [m2[m] for m in CORE if m in m2] + [d3[m] for m in CORE if m in d3]
    got = nfit.common_theta0(pool, i_tol=itol)
    if not got:
        print(f'   I_TOL={itol}: no common range'); continue
    T0, _ = got
    out = {}
    for tag, C in (('2D', m2), ('3D', d3)):
        for mg in CORE:
            if mg in C:
                r = nfit.fit_local_sys(C[mg], T0, i_tol=itol)
                if r:
                    out[(tag, mg)] = r
    a = out.get(('2D', 0.0)); b = out.get(('3D', 0.0))
    p2 = max((v['n'] for (t, m), v in out.items() if t == '2D'), default=np.nan)
    p3 = max((v['n'] for (t, m), v in out.items() if t == '3D'), default=np.nan)
    ks = np.mean([v['k'] for v in out.values()])
    print(f'   I_TOL={itol:<5g} Theta_0={T0:.3e}  <k>={ks:4.1f}   '
          f'n_2D(0)={a["n"]:+.4f}+/-{a["tot"]:.4f}  n_3D(0)={b["n"]:+.4f}'
          f'+/-{b["tot"]:.4f}   peak2D={p2:.4f} peak3D={p3:.4f}')
for ptol in (0.05, 0.10, 0.15, 0.30, 1.0):
    pool = [m2[m] for m in CORE if m in m2] + [d3[m] for m in CORE if m in d3]
    got = nfit.common_theta0(pool, p_tol=ptol)
    if not got:
        continue
    T0, _ = got
    a = nfit.fit_local_sys(m2[0.0], T0, p_tol=ptol)
    b = nfit.fit_local_sys(d3[0.0], T0, p_tol=ptol)
    print(f'   P_TOL={ptol:<5g} Theta_0={T0:.3e}   n_2D(0)={a["n"]:+.4f}'
          f'+/-{a["tot"]:.4f}  n_3D(0)={b["n"]:+.4f}+/-{b["tot"]:.4f}')
