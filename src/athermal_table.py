"""Table IX: athermal against thermostatted mu at matched Theta, and the shift.

WHY THIS EXISTS.  analyze_nothermo.py prints the athermal cells, and it
overlays the thermostatted curve for the two frictions sweep_restit happens to
carry.  The manuscript's table runs over six frictions and quotes a percentage
shift, and nothing printed that column: the thermostatted values and the
percentages were arithmetic done by hand on fits made elsewhere.  Every error
found in this project came from that category, so this script does it.

WHAT IT COMPARES.  For each grain friction, the athermal cell of
sweep_nothermo2 (no bath, e = 0.5, gdot = 1e-3) gives Theta_shear and mu.  The
thermostatted curve for the same friction in sweep_steady2d is fitted over its
gated setpoints and evaluated at that same Theta, with the error from the fit.
The two therefore differ in the presence of the bath and in nothing else that
the scan controls.

Usage:  python3 athermal_table.py     (run from the directory holding sweeps)
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

GD = 1.0e-3
E_ATH = 0.5
ZMIN = 3.0
RX_NT = (r'log\.(?P<kind>nt|ntk)_g(?P<g>[0-9.eE+-]+)_e(?P<e>[0-9.]+)'
         r'_mu(?P<mu>[0-9.]+)_s(?P<s>\d+)$')
RX_M = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'


def athermal():
    out = {}
    for p in sorted(glob.glob('sweep_nothermo2/log.*')):
        m = re.match(RX_NT, os.path.basename(p))
        if not m or m.group('kind') != 'nt':
            continue
        if abs(float(m.group('g')) - GD) > 1e-12:
            continue
        if abs(float(m.group('e')) - E_ATH) > 1e-9:
            continue
        q = nfit.parse_log(p)
        if q:
            out.setdefault(float(m.group('mu')), []).append(q)
    return out


def thermostatted():
    out = {}
    for p in glob.glob('sweep_steady2d/log.mu*'):
        m = re.match(RX_M, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('mu')), []).append(q)
    return out


def gated(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    keep = [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]
    return keep, byT


def at_theta(runs, theta):
    """Thermostatted mu at one temperature, from a local quadratic, with error.

    The quadratic is fitted in u = ln(Theta/theta) so that the value wanted is
    the intercept and its error comes straight out of the covariance matrix.
    """
    keep, byT = gated(runs)
    sel = [x for t in keep for x in byT[t] if x['Theta'] > 0 and x['mu'] > 0]
    if len(sel) < 4:
        return None
    u = np.log([x['Theta'] for x in sel]) - np.log(theta)
    y = np.log([x['mu'] for x in sel])
    A = np.vstack([np.ones_like(u), u, u ** 2]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    cov = (float(res @ res) / max(len(u) - 3, 1)) * np.linalg.inv(A.T @ A)
    mu = float(np.exp(c[0]))
    return mu, mu * float(np.sqrt(cov[0, 0])), (float(min(u)), float(max(u)))


def main():
    A, T = athermal(), thermostatted()
    if not A:
        print('sweep_nothermo2 is empty; nothing to compare.')
        return 1
    print('=' * 96)
    print('ATHERMAL AGAINST THERMOSTATTED mu AT MATCHED Theta'
          f'   (2D, e = {E_ATH}, gdot = {GD:.0e})')
    print('=' * 96)
    print(f'  {"mu_g":>6}{"Th_sh/(gd)^2":>14}{"Z":>7}{"mu athermal":>13}'
          f'{"mu thermostatted":>20}{"Delta":>10}{"extrapolation in ln Th":>24}')
    for mg in sorted(A):
        r = A[mg]
        th = float(np.mean([x['Theta'] for x in r]))
        mu_a = float(np.mean([x['mu'] for x in r]))
        z = float(np.mean([x.get('Z', np.nan) for x in r]))
        if mg not in T:
            print(f'  {mg:>6g}{th/GD**2:>14.0f}{z:>7.2f}{mu_a:>13.4f}'
                  f'{"no thermostatted cell":>20}')
            continue
        got = at_theta(T[mg], th)
        if got is None:
            print(f'  {mg:>6g}{th/GD**2:>14.0f}{z:>7.2f}{mu_a:>13.4f}'
                  f'{"too few setpoints":>20}')
            continue
        mu_t, err, (lo, hi) = got
        pct = 100.0 * (mu_a / mu_t - 1.0)
        inside = 'interior' if lo <= 0.0 <= hi else 'OUTSIDE the fitted range'
        print(f'  {mg:>6g}{th/GD**2:>14.0f}{z:>7.2f}{mu_a:>13.4f}'
              f'{f"{mu_t:.4f} +/- {err:.4f}":>20}{pct:>+9.1f}%'
              f'{inside:>24}')
    print('\n  Delta is (athermal - thermostatted)/thermostatted, the column the')
    print('  manuscript quotes.  A positive value means removing the bath makes')
    print('  the packing stronger at the same temperature.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
