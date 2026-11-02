"""Sensitivity of n to the pressure and inertial-number cutoffs.

The barostat cutoff (std(P)/mean(P) <= 0.15) and the fixed-I cutoff
(|<I>/median - 1| <= 0.03) were set without a recorded derivation. This script
refits n at the canonical Theta_0 for every case of the two- and
three-dimensional grain-friction scans with each threshold moved, and reports
the largest shift in units of the quoted error. Run from RESEARCH_LAMMPS/.

Result (2026-11-02): pressure limit 0.10 to 0.30 and inertial-number limit
0.02 to 0.06 change no value of n in either scan. Comparing the surviving
setpoints directly, every pressure limit from 0.05 upward (tested to 10) and
every inertial-number limit from 0.02 to 0.06 keeps exactly the same setpoints
in both scans; tighter or looser limits start to change them. Every setpoint the fixed-I
cutoff removes is the coldest of its case, 3% to 50% off the median, and the
setpoints near either threshold are also removed by the other cutoffs.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import nfit  # noqa: E402

RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
CANON = 8.4664e-04
SCANS = (('sweep_steady2d', 3), ('sweep_steady3d', 4))
VARIANTS = ((0.03, 0.10), (0.03, 0.20), (0.03, 0.30), (0.02, 0.15), (0.06, 0.15))


def cells(d):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(RXM, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('mu')), []).append(q)
    return out


def nval(runs, zmin, i_tol, p_tol):
    Ts, _ = nfit.surviving_setpoints(runs, i_tol=i_tol, p_tol=p_tol, z_min=zmin)
    if len(Ts) < 4:
        return None
    r = nfit.fit_local_sys(runs, CANON, fn=nfit.fit_local_adaptive,
                           restrict_to=Ts, z_min=zmin, i_tol=i_tol, p_tol=p_tol)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


for d, zmin in SCANS:
    C = cells(d)
    for i_tol, p_tol in VARIANTS:
        shifts = []
        for mu, runs in sorted(C.items()):
            base = nval(runs, zmin, nfit.I_TOL, nfit.P_TOL)
            v = nval(runs, zmin, i_tol, p_tol)
            if base and v:
                shifts.append(abs(v[0] - base[0]) / base[1])
            elif base and not v:
                shifts.append(np.inf)
        print(f'{d}  I limit {i_tol:.2f}  P limit {p_tol:.2f}  '
              f'largest shift {max(shifts):.3f} errors over {len(shifts)} cases')
