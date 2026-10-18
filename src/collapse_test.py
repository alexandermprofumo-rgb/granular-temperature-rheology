"""The three shear-rate series in Kim and Kamrin's own coordinates.

WHY THIS EXISTS.  The Letter's abstract says no constant exponent restores the
collapse of mu Theta^p = f(I).  The separability test in the companion makes
that case with a likelihood ratio.  This script makes it the way the field
reads it: choose the single p that best collapses the series at one friction,
and ask how much scatter survives at each inertial number.  Under Eq. (1) the
points of one series share one I and so must share one value of mu Theta^p.

It also prices the failure.  A law calibrated at Theta_0 with p = 1/6 predicts
mu(Theta) = mu(Theta_0) (Theta/Theta_0)^(-1/6) at every shear rate; the script
reports how far the measured mu departs from that across the shared window.

The series, the shared window and the jamming gate are exactly those of
Fig. 1 of the Letter (make_letter_figures.py, panel_arms).

Usage:  python3 collapse_test.py     (from the directory holding the sweeps)
Writes collapse_test.csv, one row per setpoint, for the figure.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

ZMIN = 3.0
CANON = 8.4664e-04
P_KK = 1 / 6
P_KK2D = 1 / 8     # Kim and Kamrin's exponent for disks; these series are 2D
RXG = (r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
       r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
ARMS = (('slow', 1.0e-4, 'sweep_iscan2/log.g3.162e-4_mu*', RXG),
        ('mid', 3.162e-4, 'sweep_steady2d/log.mu*', RXM),
        ('fast', 1.0e-3, 'sweep_iscan2/log.g3.162e-3_mu*', RXG))


def arm_cells(pattern, rx):
    out = {}
    for key, runs in nfit.load_sweep(pattern, rx).items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


def jammed(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def series(mg):
    """Per arm: arrays of Theta, mu, I, one point per setpoint in the window."""
    pools = [(tag, DATA[tag][mg]) for tag, _, _, _ in ARMS if mg in DATA[tag]]
    T0, (lo, hi) = nfit.common_theta0([p[1] for p in pools])
    # the reference temperature Fig. 1 uses: the canonical one where it lies
    # well inside the shared window, the window's own centre otherwise
    global TUSE
    TUSE = CANON if lo * 1.3 <= CANON <= hi / 1.3 else T0
    out = {}
    for tag, runs in pools:
        keep = [t for t in jammed(runs, ZMIN)
                if all(lo * 0.999 <= r['Theta'] <= hi * 1.001
                       for r in nfit.surviving_setpoints(
                           runs, z_min=ZMIN, restrict_to=[t])[1].get(t, [])
                       if r['Theta'] > 0)]
        _, byT = nfit.surviving_setpoints(runs, z_min=ZMIN, restrict_to=keep)
        pts = []
        for t in keep:
            sel = [r for r in byT.get(t, []) if r['Theta'] > 0 and r['mu'] > 0]
            if sel:
                pts.append([np.mean([r[k] for r in sel])
                            for k in ('Theta', 'mu', 'I')])
        out[tag] = np.array(sorted(pts)).T
    return out, (lo, hi)


def collapse_rms(arms, p):
    """RMS scatter of ln(mu Theta^p) about each series' own mean."""
    res = [np.log(mu) + p * np.log(th) for th, mu, _ in arms.values()]
    res = np.concatenate([r - r.mean() for r in res])
    return float(np.sqrt(np.mean(res ** 2)))


def best_p(arms):
    """The common slope that minimizes the scatter, series intercepts free."""
    x = np.concatenate([np.log(th) - np.log(th).mean() for th, _, _ in arms.values()])
    y = np.concatenate([np.log(mu) - np.log(mu).mean() for _, mu, _ in arms.values()])
    return float(-(x @ y) / (x @ x))


def floor_rms(arms):
    """Scatter left when every series gets its own quadratic in ln Theta,
    an estimate of what noise and curvature alone leave."""
    res = []
    for th, mu, _ in arms.values():
        u = np.log(th)
        c = np.polyfit(u, np.log(mu), 2)
        res.append(np.log(mu) - np.polyval(c, u))
    res = np.concatenate(res)
    dof = len(res) - 3 * len(arms)
    return float(np.sqrt(np.sum(res ** 2) / dof))


def kk_error(arms, theta0, p=P_KK):
    """Largest departure of measured mu from a p = 1/6 law calibrated at theta0."""
    worst = {}
    for tag, (th, mu, _) in arms.items():
        u = np.log(th / theta0)
        c = np.polyfit(u, np.log(mu), min(2, len(u) - 1))
        mu0 = np.exp(np.polyval(c, 0.0))
        pred = mu0 * (th / theta0) ** (-p)
        rel = mu / pred - 1.0
        k = int(np.argmax(np.abs(rel)))
        worst[tag] = (float(rel[k]), float(th[k] / theta0))
    return worst


DATA = {tag: arm_cells(p, rx) for tag, _, p, rx in ARMS}

rows = []
for mg in (0.0, 0.15, 0.3):
    arms, (lo, hi) = series(mg)
    p_star = best_p(arms)
    r_star, r_kk, r_floor = (collapse_rms(arms, p_star), collapse_rms(arms, P_KK),
                             floor_rms(arms))
    print(f'mu_g = {mg}:  window {lo:.3g} to {hi:.3g} ({hi / lo:.1f}x),'
          f' calibrated at {TUSE:.3g}')
    print(f'  best common p = {p_star:.3f}')
    print(f'  scatter of ln(mu Theta^p) about each series:'
          f'  {100 * r_star:.2f}% at best p,  {100 * r_kk:.2f}% at p = 1/6,'
          f'  {100 * r_floor:.2f}% with each series fitted alone')
    print(f'  ratio best-p scatter / own-fit scatter = {r_star / r_floor:.2f}')
    for tag, (rel, x) in kk_error(arms, TUSE).items():
        print(f'  p = 1/6 calibrated at Theta_0, {tag}: worst {100 * rel:+.1f}%'
              f' at Theta/Theta_0 = {x:.2f}')
    for tag, (rel, x) in kk_error(arms, TUSE, P_KK2D).items():
        print(f'  p = 1/8 calibrated at Theta_0, {tag}: worst {100 * rel:+.1f}%'
              f' at Theta/Theta_0 = {x:.2f}')
    # the cost of the form itself: even the best single exponent misses
    for tag, (rel, x) in kk_error(arms, TUSE, p_star).items():
        print(f'  best p calibrated at Theta_0, {tag}: worst {100 * rel:+.1f}%'
              f' at Theta/Theta_0 = {x:.2f}')
    for tag, (th, mu, I) in arms.items():
        for a, b, c in zip(th, mu, I):
            rows.append([mg, tag, a, b, c, b * a ** p_star, p_star])

with open('collapse_test.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mu_g', 'arm', 'Theta', 'mu', 'I', 'mu_Theta_pstar', 'p_star'])
    w.writerows(rows)
print('wrote collapse_test.csv')
