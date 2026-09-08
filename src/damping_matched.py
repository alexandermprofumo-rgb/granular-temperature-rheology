"""Is the stiffness dependence of n a contact-time effect or real elasticity?

WHY THIS EXISTS.  stiffness_matched.py finds dn/dlnE = +0.0360 +/- 0.0049 at
mu_g = 0.3, surviving the jamming gate and matched Theta windows, and
analyze_collapse_pooled.py shows the effect does not act through coordination:
the stiffness points miss the n(dZ_eff) curve traced by friction at 4.7 sigma
RMS.  That leaves one named candidate, contact duration relative to the bath.

THE TEST IS QUANTITATIVE, NOT QUALITATIVE.  Both levers move the same ratio,
so the timescale hypothesis predicts a specific number rather than merely
"some dependence".

  Hertzian contact time goes as t_c ~ E^(-2/5) at fixed pressure, and the
  bath time is t_damp, so with r = t_c/t_damp:

      dln r / dlnE      = -2/5          (stiffen at fixed bath)
      dln r / dln t_damp = -1           (slow the bath at fixed stiffness)

  If n depends on E and t_damp ONLY through r, with dn/dln r = s, then

      dn/dlnE          = -0.4 s
      dn/dlnPi_damp    = -1.0 s   =>   dn/dlnPi_damp = 2.5 x dn/dlnE.

  So the measured stiffness slope predicts dn/dlnPi_damp = +0.090 +/- 0.012 at
  mu_g = 0.3.  Anything near zero refutes the timescale account and leaves the
  stiffness dependence as a property of the contact elasticity itself.

Pi_damp = t_damp sqrt(P/rho)/d is fixed at fixed pressure, so along this lever
lnPi_damp and ln t_damp differ by a constant and the slope is the same in
either.  n is computed exactly as in the paper: jamming gate at Z >= D+1,
adaptive-degree local fit at the canonical Theta_0, calibrated jackknife with
the degree systematic folded in, and then again on the setpoints common to
every damping value so no cell is compared over a different Theta range.

Usage:  python3 damping_matched.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

RX = (r'log\.td(?P<td>[0-9.]+)_mu(?P<mu>[0-9.]+)'
      r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
DIR, ZMIN = 'sweep_lev_td', 3.0
CANON = 8.4664e-04

# from stiffness_matched.py, 2D, matched windows
DNDLNE = {0.1: (-0.0126, 0.0059), 0.3: (+0.0360, 0.0049), 1.0: (+0.0061, 0.0099)}
RATIO = 2.5                      # (dln r/dln t_damp) / (dln r/dlnE) = -1 / -0.4

# check_pidamp.py, which produces the Sec. IV E numbers, evaluates n at a
# per-friction Theta_0 taken from that cell's common range; this script uses
# the canonical Theta_0 throughout so the damping slope is measured on the same
# footing as the stiffness slope. The two therefore need not agree exactly, and
# the paper's values are carried here so the verdict can be checked under both.
PAPER_TD = {0.1: (-0.0102, 0.0075), 0.3: (-0.0081, 0.0088), 1.0: (+0.0084, 0.0055)}


def cells():
    out = {}
    for p in glob.glob(f'{DIR}/log.td*'):
        m = re.match(RX, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault((float(m.group('mu')), float(m.group('td'))), []).append(q)
    return out


def gated(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def n_of(runs, keep):
    r = nfit.fit_local_sys(runs, CANON, fn=nfit.fit_local_adaptive,
                           restrict_to=keep, z_min=ZMIN)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


def wslope(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    return (cov @ A.T @ np.diag(w) @ y)[1], np.sqrt(cov[1, 1])


def main():
    C = cells()
    if not C:
        print(f'{DIR} not unpacked; run this beside the log archive.')
        return 1
    print('=' * 86)
    print('THE DAMPING LEVER AT FIXED STIFFNESS, PAPER ESTIMATOR')
    print('=' * 86)
    print('  timescale hypothesis: dn/dlnPi_damp = 2.5 x dn/dlnE, same sign.\n')
    for mg in sorted({k[0] for k in C}):
        tds = sorted(td for (m, td) in C if m == mg)
        keep, ok = {}, []
        for td in tds:
            k = gated(C[(mg, td)])
            if len(k) >= 4:
                keep[td], _ = k, ok.append(td)
        if len(ok) < 3:
            print(f'  mu_g={mg:g}: too few fittable damping values\n')
            continue
        common = set(keep[ok[0]])
        for td in ok[1:]:
            common &= set(keep[td])
        print(f'  mu_g={mg:g}   {len(ok)} damping values, '
              f'{len(common)} setpoints common to all')
        for label, sets in (('unmatched', {t: keep[t] for t in ok}),
                            ('matched', {t: sorted(common) for t in ok})):
            if label == 'matched' and len(common) < 4:
                print('     matched: fewer than four common setpoints')
                continue
            xs, ys, es = [], [], []
            for td in ok:
                got = n_of(C[(mg, td)], sets[td])
                if got:
                    xs.append(np.log(td)); ys.append(got[0]); es.append(got[1])
            if len(xs) < 3:
                print(f'     {label:>9}: too few cells survive')
                continue
            s, se = wslope(np.array(xs), np.array(ys), np.array(es))
            print(f'     {label:>9}: n = ' + '  '.join(f'{v:.3f}' for v in ys))
            print(f'     {"":>9}  dn/dlnPi_damp = {s:+.4f} +/- {se:.4f}'
                  f'  ({abs(s)/se:.1f} sigma from zero)')
            if label == 'matched' and mg in DNDLNE:
                p, pe = DNDLNE[mg]
                pred, prede = RATIO * p, RATIO * pe
                gap = abs(s - pred) / float(np.hypot(se, prede))
                print(f'     {"":>9}  timescale predicts {pred:+.4f} +/- '
                      f'{prede:.4f} from dn/dlnE = {p:+.4f}')
                print(f'     {"":>9}  measured minus predicted: {gap:.1f} sigma'
                      f'   -> {"CONSISTENT" if gap < 2 else "REFUTED"}')
                if mg in PAPER_TD:
                    q, qe = PAPER_TD[mg]
                    g2 = abs(q - pred) / float(np.hypot(qe, prede))
                    print(f'     {"":>9}  same verdict under the Sec. IV E value '
                          f'{q:+.4f} +/- {qe:.4f}: {g2:.1f} sigma'
                          f'   -> {"CONSISTENT" if g2 < 2 else "REFUTED"}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
