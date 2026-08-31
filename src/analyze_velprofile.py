"""Is the streaming profile affine? -- the assumption behind every Theta here.

compute temp/deform subtracts the affine field v_x = gdot*y and calls the
remainder the granular temperature. If the real profile is curved or banded,
the unremoved mean flow is counted as Theta, and since n is a Theta-derivative
every constraint in the project inherits the error. No sweep dumps per-atom
velocities, so this could not be checked on existing data; sweep_velprof adds
a binned v_x(y), v_y(y) profile to runs that are otherwise identical to
sweep_steady2d.

STATISTIC. Take the last averaging block (the measurement window), fit the bin
means of v_x against reduced y, and report

  * the RMS residual of that fit, as a fraction of the fitted velocity range;
  * Theta_spurious = <v_resid^2>/2 over bins, against the Theta the log reports.

PASS: Theta_spurious / Theta < 0.05 in every cell -- i.e. under 5% of the
reported granular temperature is unremoved mean flow.

Usage:  python3 analyze_velprofile.py
"""
import glob
import os
import re

import numpy as np

import nfit

RX = r'prof\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'


def last_block(path):
    """The final timestep block: (coord, vx, vy, ncount) arrays."""
    rows, cur = [], []
    for line in open(path):
        if line.startswith('#'):
            continue
        t = line.split()
        if len(t) == 3:                     # timestep header
            if cur:
                rows = cur
            cur = []
        elif len(t) >= 6:
            cur.append([float(x) for x in t])
    if cur:
        rows = cur
    if not rows:
        return None
    a = np.array(rows)
    return a[:, 1], a[:, 3], a[:, 4], a[:, 2]


def main():
    paths = sorted(glob.glob('sweep_velprof/prof.*'))
    if not paths:
        print('sweep_velprof is empty -- run ./run_velprofile.sh first.')
        return
    print('=' * 84)
    print('VELOCITY PROFILE   is the streaming field affine, as temp/deform assumes?')
    print('=' * 84)
    print(f'  {"mu_g":>6}{"Tgran":>9}{"seed":>5}{"bins":>6}{"lin R^2":>9}'
          f'{"resid/range":>13}{"Theta_spur":>12}{"Theta(log)":>12}{"ratio":>9}')
    bad = []
    for p in paths:
        m = re.match(RX, os.path.basename(p))
        if not m:
            continue
        got = last_block(p)
        if got is None:
            continue
        y, vx, vy, nc = got
        ok = nc > 0
        y, vx, vy = y[ok], vx[ok], vy[ok]
        if len(y) < 6:
            continue
        A = np.vstack([np.ones_like(y), y]).T
        c, *_ = np.linalg.lstsq(A, vx, rcond=None)
        res = vx - A @ c
        rng = float(np.ptp(A @ c))
        ss = float(np.sum((vx - vx.mean()) ** 2))
        r2 = 1.0 - float(res @ res) / ss if ss > 0 else np.nan
        th_sp = float(np.mean(res ** 2 + (vy - vy.mean()) ** 2) / 2)
        lab = os.path.basename(p).replace('prof.', '')
        q = nfit.parse_log(f'sweep_velprof/log.{lab}')
        th = q['Theta'] if q else np.nan
        ratio = th_sp / th if th and np.isfinite(th) else np.nan
        if np.isfinite(ratio) and ratio >= 0.05:
            bad.append((lab, ratio))
        print(f'  {float(m.group("mu")):>6g}{float(m.group("T")):>9g}'
              f'{m.group("s"):>5}{len(y):>6}{r2:>9.4f}'
              f'{np.sqrt(res@res/len(res))/max(rng,1e-30):>13.4f}'
              f'{th_sp:>12.3e}{th:>12.3e}{ratio:>9.3f}')
    print()
    if bad:
        print(f'  FAIL: {len(bad)} cells have >5% of Theta as unremoved mean flow:')
        for lab, r in bad:
            print(f'     {lab}  {r:.1%}')
        print('  Every n in the project inherits this. It is a correction to Theta,')
        print('  not a correction to mu, so it shifts the Theta axis and therefore n.')
    else:
        print('  PASS: unremoved mean flow is under 5% of Theta in every cell, so')
        print('  temp/deform is measuring what the theory calls Theta.')


if __name__ == '__main__':
    main()
