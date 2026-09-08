"""What do the pressure and tangential-stiffness levers do to n itself?

WHY THIS EXISTS.  The protocol treats P and k_t as LEVERS: knobs turned so that
a candidate state variable can be pushed sideways and its residuals tested.
Nothing in the plan ever asks what the lever does to n, because in that design
the lever is the control rather than the treatment.  stiffness_matched.py found
that the E lever moves n by 7.3 sigma at mu_g = 0.3, so the same question is
worth putting to the other two.

THE PRESSURE LEVER CARRIES A PREDICTION.  Its driver scales gdot as sqrt(P) to
hold I fixed, so only kappa = E/P moves along it.  If n depends on the
dimensionless kappa rather than on E as a number in an input deck, then

    dn/dlnP = -dn/dlnE = -0.036 +/- 0.005   at mu_g = 0.3.

A confirmation makes the stiffness result a dependence on a dimensionless
group.  A null makes n depend on E and P separately, which would mean kappa is
not the variable and the E result needs a different account.  The pressure
lever also reaches mu_g = 0.05, 0.1, 0.15 and 0.2, so unlike the E lever it
samples the rising branch where the friction law is fitted.

THE TANGENTIAL LEVER CARRIES NONE.  k_t/k_n is its own dimensionless group; it
is tested here only because it has the same never-asked status.

THE TWO LEVERS NEED DIFFERENT REFERENCE TEMPERATURES, and getting this wrong
inverts the pressure answer.  Along the pressure lever the measured Theta
scales with P: the shared setpoints span [9.0e-5, 1.0e-3] at P = 5 and
[1.1e-3, 1.3e-2] at P = 50.  The cells are therefore disjoint in absolute
Theta, and evaluating them all at one absolute Theta_0 puts the high-pressure
cells outside their own data, which is extrapolation from a polynomial rather
than measurement.  A first version of this script did exactly that and returned
n = -0.231 at P = 50.  The reference temperature is scaled as Theta_0 P/P_ref
instead, so every cell is read at the same dimensionless temperature and inside
its own range.  The tangential lever needs no such treatment: its cells overlap
in Theta already, and the canonical Theta_0 is used unchanged.  The measured
range at the shared setpoints is printed for both so the choice can be checked
rather than trusted.

Usage:  python3 levers_matched.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

# name, directory, filename pattern, z_min, x label, scale Theta_0 with the
# lever, and the sign converting dn/dln(lever) into dn/dln(kappa).  kappa = E/P,
# so the modulus lever carries +1, the pressure lever -1, and the tangential
# lever does not move kappa at all.
LEVERS = (
    ('modulus', 'sweep_lev_E',
     r'log\.E(?P<v>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     3.0, 'lnE', False, +1),
    ('pressure', 'sweep_lev_P',
     r'log\.P(?P<v>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     3.0, 'lnP', True, -1),
    ('tangential', 'sweep_lev_kt',
     r'log\.k(?P<v>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     3.0, 'ln k_t', False, 0),
)
CANON = 8.4664e-04
# stiffness_matched.py, 2D, matched windows
DNDLNE = {0.1: (-0.0126, 0.0059), 0.3: (+0.0360, 0.0049), 1.0: (+0.0061, 0.0099)}


def cells(d, rx):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault((float(m.group('mu')), float(m.group('v'))), []).append(q)
    return out


def gated(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def theta_at(runs, keep, zmin):
    _, byT = nfit.surviving_setpoints(runs, z_min=zmin, restrict_to=keep)
    th = [r['Theta'] for t in keep for r in byT[t] if r['Theta'] > 0]
    return (min(th), max(th)) if th else None


def n_of(runs, keep, zmin, theta0):
    r = nfit.fit_local_sys(runs, theta0, fn=nfit.fit_local_adaptive,
                           restrict_to=keep, z_min=zmin)
    if not r:
        return None
    return r['n'], float(np.hypot(r['tot'], r.get('sys_deg', 0.0)))


def wslope(x, y, e):
    w = 1.0 / np.asarray(e) ** 2
    A = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    return (cov @ A.T @ np.diag(w) @ y)[1], np.sqrt(cov[1, 1])


def main():
    kappa = {}
    for name, d, rx, zmin, xlab, scale_t0, ksign in LEVERS:
        C = cells(d, rx)
        if not C:
            print(f'{d} not unpacked; skipping {name}.\n')
            continue
        print('=' * 90)
        print(f'THE {name.upper()} LEVER, PAPER ESTIMATOR')
        print('=' * 90)
        for mg in sorted({k[0] for k in C}):
            vs = sorted(v for (m, v) in C if m == mg)
            keep, ok = {}, []
            for v in vs:
                k = gated(C[(mg, v)], zmin)
                if len(k) >= 4:
                    keep[v] = k
                    ok.append(v)
            if len(ok) < 3:
                print(f'  mu_g={mg:g}: too few fittable lever values\n')
                continue
            common = set(keep[ok[0]])
            for v in ok[1:]:
                common &= set(keep[v])
            print(f'  mu_g={mg:g}   lever values {ok}, '
                  f'{len(common)} setpoints common to all')
            for v in ok:
                sp = theta_at(C[(mg, v)], sorted(common), zmin)
                if sp:
                    t0 = CANON * (v / 10.0) if scale_t0 else CANON
                    inside = 'inside' if sp[0] <= t0 <= sp[1] else 'OUTSIDE'
                    print(f'     v={v:<8g} Theta over shared setpoints '
                          f'[{sp[0]:.3e}, {sp[1]:.3e}]  '
                          f'Theta_0={t0:.3e} ({inside})')
            if len(common) < 4:
                print('     fewer than four common setpoints; no matched '
                      'comparison.\n')
                continue
            xs, ys, es = [], [], []
            for v in ok:
                t0 = CANON * (v / 10.0) if scale_t0 else CANON
                got = n_of(C[(mg, v)], sorted(common), zmin, t0)
                if got:
                    xs.append(np.log(v)); ys.append(got[0]); es.append(got[1])
            if len(xs) < 3:
                print('     too few cells survive\n')
                continue
            s, se = wslope(np.array(xs), np.array(ys), np.array(es))
            print('       matched: n = ' + '  '.join(f'{v:.3f}' for v in ys))
            print(f'                dn/d{xlab} = {s:+.4f} +/- {se:.4f}'
                  f'  ({abs(s)/se:.1f} sigma from zero)')
            if ksign:
                kappa.setdefault(mg, {})[name] = (ksign * s, se)
                print(f'                dn/dln(kappa) = {ksign*s:+.4f}'
                      f' +/- {se:.4f}')
            print()

    print('=' * 90)
    print('THE TWO KAPPA LEVERS COMBINED')
    print('=' * 90)
    print('  kappa = E/P, so the modulus and pressure levers move it in')
    print('  opposite directions and are independent measurements of the same')
    print('  derivative.  Where both exist they are checked against each other')
    print('  before being pooled; a disagreement would mean n depends on E and')
    print('  P separately and kappa is not the variable.\n')
    print(f'  {"mu_g":>6}{"modulus":>20}{"pressure":>20}{"agree":>9}'
          f'{"dn/dln(kappa)":>20}')
    for mg in sorted(kappa):
        got = kappa[mg]
        cols = []
        for nm in ('modulus', 'pressure'):
            cols.append(f'{got[nm][0]:+.4f}+/-{got[nm][1]:.4f}'
                        if nm in got else '-')
        if len(got) == 2:
            (a, ae), (b, be) = got['modulus'], got['pressure']
            agree = abs(a - b) / float(np.hypot(ae, be))
            w = np.array([1 / ae ** 2, 1 / be ** 2])
            m = float(np.dot(w, [a, b]) / w.sum())
            me = float(1 / np.sqrt(w.sum()))
            ag = f'{agree:.1f}s'
        else:
            (m, me), = got.values()
            ag = '-'
        print(f'  {mg:>6g}{cols[0]:>20}{cols[1]:>20}{ag:>9}'
              f'{f"{m:+.4f}+/-{me:.4f}":>20}')
    print('\n  A sign change in the last column locates the peak of n(mu_g):')
    print('  stiffening raises n below it and lowers n above it.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
