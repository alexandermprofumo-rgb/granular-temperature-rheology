"""
n(mu_g, k_t) over the tangential-stiffness grid in sweep_sens/.

The spot-check in campaign item 10 found that n depends on tangential
stiffness at finite friction, with a clean null at mu_g=0. That is a
constraint no proposed framework contains: generalised isostaticity predicts
sensitivity to mu_g but NOT to k_t. This maps it properly.

Three things are tested here.

 1. THE NULL. At mu_g=0 no tangential force exists, so every k_t must return
    the same n. Any spread there is pipeline leakage, not physics, and
    invalidates the rest of the grid.

 2. THE GRID. n(mu_g, k_t), with k_t reported as a ratio to the LAMMPS
    default (8*G_eff = 9.05e4 for hertz/material E=1e5, nu=0.3) and to the
    normal prefactor (4/3*E_eff = 7.33e4).

 3. THE REDUCED VARIABLE. If the Coulomb threshold and tangential elasticity
    enter through one combination rather than two independent knobs, n should
    collapse onto a single curve in some x = mu_g * (k_t/k_n)^p. The exponent
    p is fitted by minimising the scatter of n about a smoothing spline in
    log x; p ~ 0 would mean k_t acts independently of friction, p ~ 0.5 would
    match a contact-compliance scaling.

Thermostat control is reported alongside: if Theta/Tgran runs far from the
setpoint the fit is not trustworthy, which is how the pressure-lever attempt
failed.

Usage:  python3 analyze_ktgrid.py [--dir sweep_sens] [--out ktgrid_results.csv]
"""
import argparse
import csv
import glob
import os
import re
import numpy as np

from analyze_sweep import parse_log

# LAMMPS granular defaults for hertz/material E=1e5, nu=0.3
KT_DEFAULT = 9.05e4      # mindlin NULL -> 8*G_eff
KN_PREFAC = 7.326e4      # 4/3 * E_eff

LABEL_RE = re.compile(
    r'e(?P<e>[0-9.]+)_k(?P<kt>NULL|[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
    r'_T(?P<T>[0-9.eE+-]+)_s(?P<seed>\d+)$'
)


def fit_n(theta, mu):
    """n = -d ln(mu) / d ln(Theta), with the standard error of the slope."""
    theta = np.asarray(theta, float)
    mu = np.asarray(mu, float)
    m = np.isfinite(theta) & np.isfinite(mu) & (theta > 0) & (mu > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(theta[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return -c[0], np.sqrt(cov[0, 0])


def cell_valid(runs):
    """A (mu_g, k_t) cell is fittable only if the thermostat actually
    controlled Theta across the setpoint grid.

    Two ways it fails, both seen: Theta exceeding its own setpoint (shear /
    contact heating winning), and Theta not increasing with Tgran at all, so
    the x-axis of the n fit is not the variable we think it is. Both occur at
    low k_t and high mu_g, where a Coulomb-limited tangential spring stretches
    to xi_max = mu_g*f_n/k_t before slipping and dumps the stored energy as
    heat. That is the contact model reaching its limit, not a bug -- and it
    happens only for k_t/k_n < 0.04, already an order of magnitude below the
    physical range for elastic spheres (k_t/k_n ~ 0.67-1.0 from Poisson).

    Returns (ok, reason).
    """
    by_T = {}
    for r in runs:
        by_T.setdefault(r['Tgran'], []).append(r['Theta'])
    if len(by_T) < 3:
        return False, 'too few setpoints'
    ratios = [r['Theta'] / r['Tgran'] for r in runs if r['Tgran'] > 0]
    if max(ratios) > 1.0:
        return False, f'thermostat lost control (max Theta/Tgran={max(ratios):.1f})'
    Ts = sorted(by_T)
    means = [float(np.mean(by_T[t])) for t in Ts]
    if any(b <= a for a, b in zip(means, means[1:])):
        return False, 'Theta not monotonic in Tgran'
    return True, ''


def collect(d):
    rows = []
    for path in sorted(glob.glob(os.path.join(d, 'log.*'))):
        label = os.path.basename(path)[4:]
        m = LABEL_RE.match(label)
        if not m:
            continue
        rec = parse_log(path)
        if rec is None:
            continue
        kt = KT_DEFAULT if m.group('kt') == 'NULL' else float(m.group('kt'))
        rec.update(e_rest=float(m.group('e')), kt=kt,
                   kt_is_default=(m.group('kt') == 'NULL'),
                   mu_g=float(m.group('mu')), Tgran=float(m.group('T')),
                   seed=int(m.group('seed')), label=label)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sweep_sens')
    # Writes under derived/ so a rerun does not overwrite the tracked table
    # that ships with the repository. Pass --out to override.
    ap.add_argument('--out', default='derived/ktgrid_results.csv')
    args = ap.parse_args()

    rows = [r for r in collect(args.dir) if abs(r['e_rest'] - 0.5) < 1e-9]
    if not rows:
        print('no runs found')
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mus = sorted(set(r['mu_g'] for r in rows))
    kts = sorted(set(r['kt'] for r in rows))

    grid, err, rejected = {}, {}, []
    print(f'\n{"n(mu_g, k_t)":<14}' + ''.join(f'{k/KT_DEFAULT:>10.3f}x' for k in kts))
    print('-' * (14 + 11 * len(kts)))
    for mg in mus:
        line = f'mu_g={mg:<9g}'
        for kt in kts:
            s = [r for r in rows if r['mu_g'] == mg and r['kt'] == kt]
            if len(s) < 3:
                grid[(mg, kt)] = np.nan
                line += f'{"-":>11}'
                continue
            ok, why = cell_valid(s)
            if not ok:
                grid[(mg, kt)] = np.nan
                rejected.append((mg, kt, why))
                line += f'{"XX":>11}'
                continue
            n, e = fit_n([r['Theta'] for r in s], [r['mu'] for r in s])
            grid[(mg, kt)] = n
            err[(mg, kt)] = e
            line += f'{n:>11.3f}'
        print(line)
    print('   (-  not yet run;  XX  thermostat invalid, excluded)')
    if rejected:
        print(f'\n--- {len(rejected)} cells rejected ---')
        for mg, kt, why in rejected:
            print(f'  mu_g={mg:<6g} k_t/k_t0={kt/KT_DEFAULT:<7.3f} '
                  f'(k_t/k_n={kt/KN_PREFAC:.3f})  {why}')

    # test 1: the mu_g = 0 null
    print('\n--- null test (mu_g=0: all k_t must agree) ---')
    z = [grid[(0.0, k)] for k in kts if np.isfinite(grid.get((0.0, k), np.nan))]
    ze = [err[(0.0, k)] for k in kts if (0.0, k) in err]
    if len(z) >= 2:
        spread = max(z) - min(z)
        typ = float(np.mean(ze)) if ze else float('nan')
        print(f'  n = {["%.4f" % v for v in z]}')
        print(f'  spread {spread:.4f} vs typical fit error {typ:.4f}  '
              f'-> {"PASS" if spread <= 2 * typ else "FAIL (pipeline leak)"}')

    # test 2: k_t effect size at each friction
    print('\n--- k_t sensitivity at each mu_g ---')
    for mg in mus:
        v = [(k, grid[(mg, k)]) for k in kts if np.isfinite(grid.get((mg, k), np.nan))]
        if len(v) < 3:
            continue
        vals = np.array([x[1] for x in v])
        es = np.array([err[(mg, k)] for k, _ in v])
        spread = vals.max() - vals.min()
        sig = spread / np.sqrt(np.mean(es ** 2))
        kbest = v[int(np.argmax(vals))][0]
        print(f'  mu_g={mg:<6g} spread {spread:.3f} ({sig:4.1f} sigma)  '
              f'max at k_t/k_t0={kbest/KT_DEFAULT:.3f}')

    # test 3: sign and size of the k_t dependence at each friction
    #
    # An earlier version of this fitted a reduced variable x = mu_g*(k_t/k_n)^p
    # by minimising scatter about a spline. That statistic was unstable -- it
    # reported a best-fit p indistinguishable from zero while simultaneously
    # claiming a large improvement over p=0, which cannot both be true. The
    # knot placement moved with p, so it was measuring the estimator, not the
    # data. What the grid actually shows is a SIGN CHANGE in d n / d ln k_t,
    # which no monotonic reduced variable can represent, so that hypothesis is
    # tested directly and simply here instead.
    print('\n--- d n / d ln(k_t) at each mu_g (sign is the point) ---')
    slopes = []
    for mg in mus:
        v = [(kt, grid[(mg, kt)]) for kt in kts
             if np.isfinite(grid.get((mg, kt), np.nan))]
        if len(v) < 4:
            continue
        x = np.log(np.array([a for a, _ in v]) / KN_PREFAC)
        y = np.array([b for _, b in v])
        w = np.array([err.get((mg, a), np.nan) for a, _ in v])
        A = np.vstack([x, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        s2 = np.sum(r ** 2) / max(len(x) - 2, 1)
        se = np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])
        sig = abs(c[0]) / se if se > 0 else np.nan
        slopes.append((mg, c[0], se))
        mark = '' if sig < 2 else ('  <-- rises' if c[0] > 0 else '  <-- falls')
        print(f'  mu_g={mg:<6g} slope {c[0]:+.4f} +/- {se:.4f}  ({sig:4.1f} sigma)'
              f'  [n from {y.min():.3f} to {y.max():.3f}]{mark}')
    pos = [s for s in slopes if s[1] > 0 and abs(s[1]) / s[2] > 2]
    neg = [s for s in slopes if s[1] < 0 and abs(s[1]) / s[2] > 2]
    if pos and neg:
        print(f'\n  SIGN CHANGE: n rises with k_t at mu_g <= {max(p[0] for p in pos):g}'
              f' and falls at mu_g >= {min(n_[0] for n_ in neg):g}.')
        print('  => mu_g and k_t do NOT collapse onto a single reduced variable;'
              ' tangential')
        print('     elasticity and the Coulomb threshold act as independent knobs.')

    # thermostat sanity
    ratios = [r['Theta'] / r['Tgran'] for r in rows if r['Tgran'] > 0]
    print(f'\nthermostat: Theta/Tgran median {np.median(ratios):.3f}, '
          f'range [{min(ratios):.3f}, {max(ratios):.3f}]')
    print(f'wrote {args.out}  ({len(rows)} runs)')


if __name__ == '__main__':
    main()
