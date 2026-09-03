"""Reproduce constraint 10 -- 'n is independent of thermostat coupling'.

Built from
run_openq.sh: t_damp in {0.125 ... 4.0} (32x), mu_g in {0.1, 0.3, 1.0}, eight
Theta setpoints, three seeds, all at Pconf = 10 so Pi_damp = t_damp*sqrt(10).

This is the null that RETRACTS the first campaign's Tier 0-A, so its POWER is
the whole result.  Reported three ways: with nfit's inflated 'tot', with stat
alone, and with the calibrated systematic (jackknife excess over its own null
expectation).
"""
import numpy as np
import nfit

MUS = (0.1, 0.3, 1.0)
NREP = 120


def null_sys(runs, T0, rng):
    """Expected jackknife when the quadratic is exactly right (see jk_calib2)."""
    Ts, byT = nfit.surviving_setpoints(runs)
    sel = [x for t in Ts for x in byT[t]]
    u = np.log(np.array([x['Theta'] for x in sel]) / T0)
    y = np.log(np.array([x['mu'] for x in sel]))
    A = np.vstack([np.ones_like(u), u, u ** 2]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    s = (y - A @ c).std(ddof=3)
    out = []
    for _ in range(NREP):
        fake = [dict(x, mu=float(np.exp((A @ c)[i] + rng.normal(0, s))))
                for i, x in enumerate(sel)]
        f = nfit.fit_local_sys(fake, T0)
        if f and np.isfinite(f['sys']):
            out.append(f['sys'])
    return float(np.mean(out)) if out else np.nan


raw = nfit.load_sweep(
    'sweep_pidamp/log.td*',
    r'log\.td(?P<td>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
C = {}
for key, runs in raw.items():
    d = dict(key)
    C.setdefault((float(d['mu']), float(d['td'])), []).extend(runs)

rng = np.random.default_rng(1)
print('Pi_damp = t_damp*sqrt(10);  32x range in t_damp = 0.125 .. 4.0\n')
for mg in MUS:
    legs = {td: C[(m, td)] for (m, td) in C if m == mg}
    if len(legs) < 3:
        print(f'mu_g={mg}: {len(legs)} couplings'); continue
    got = nfit.common_theta0(list(legs.values()))
    if got is None:
        print(f'mu_g={mg}: no common Theta range'); continue
    T0, (lo, hi) = got
    xs, ns, e_tot, e_stat, e_cal = [], [], [], [], []
    print(f'mu_g = {mg}   Theta_0 = {T0:.3e}  (common range {hi/lo:.1f}x)')
    for td in sorted(legs):
        f = nfit.fit_local_sys(legs[td], T0)
        if not f:
            print(f'   t_damp={td:<6g} VOID (gated out)'); continue
        nl = null_sys(legs[td], T0, rng)
        exc = float(np.sqrt(max(f['sys'] ** 2 - nl ** 2, 0.0)))
        xs.append(np.log(td)); ns.append(f['n'])
        e_tot.append(f['tot']); e_stat.append(f['stat'])
        e_cal.append(float(np.hypot(f['stat'], exc)))
        print(f'   t_damp={td:<6g} n={f["n"]:+.4f}  stat {f["stat"]:.4f}  '
              f'sys {f["sys"]:.4f} (null {nl:.4f}, excess {exc:.4f})  '
              f'[{f["k"]} setpoints]')
    if len(xs) < 3:
        print(); continue
    x = np.array(xs)
    for nm, e in (('tot (as reported)', e_tot), ('stat only', e_stat),
                  ('calibrated', e_cal)):
        e = np.array(e)
        A = np.vstack([np.ones_like(x), x]).T
        W = np.diag(1 / e ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        c = cov @ A.T @ W @ np.array(ns)
        sl, se = float(c[1]), float(np.sqrt(cov[1, 1]))
        chi2 = float((((np.array(ns) - A @ c) / e) ** 2).sum()) / max(len(x) - 2, 1)
        # 2-sigma bound on the total variation across the measured 32x range
        span = x.max() - x.min()
        print(f'   {nm:<20} dn/dln(Pi_damp) = {sl:+.4f} +/- {se:.4f} '
              f'({abs(sl)/se:.1f}s)  chi2/dof {chi2:.2f}  '
              f'-> 2s bound on variation over the range: {2*se*span:.4f}')
    print()
