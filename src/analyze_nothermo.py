"""The athermal control: is the thermostatted mu(Theta) curve the real one?

WHY THIS EXISTS. Every Theta in this project is imposed by a Langevin bath, and
over the windows actually fitted it runs 41x to 4.1e4x the temperature the shear
itself generates (figure_numbers.json['regime']). A referee can reasonably say
the whole scan sits outside the regime where mu*Theta^n = F(I) is asserted --
that the paper measures a bath-driven material, not a sheared one. That is the
single largest scope objection to the paper and no run in the project addressed
it.

sweep_nothermo2 removes the bath entirely (in.granular_2d_nothermo_ss). Theta
settles where shear heating balances inelastic dissipation; restitution scans it
at fixed I with nothing thermostatting anything. The comparison set is
sweep_restit -- same deck family, same packings (data.re_seed*), same gdot,
same three e -- so the athermal point and the thermostatted curve are matched in
every group except the presence of the bath.

THE TEST. Fit the thermostatted mu(Theta) per (mu_g, e) cell and evaluate it at
the athermal Theta_shear. Report (a) how far that is an extrapolation, in ln
Theta, and (b) the discrepancy in sigma.

  agreement   -> the bath is a way of moving along one mu(Theta) curve, the
                 thermostatted measurement is validated, and the scope
                 objection is answered with data.
  disagreement-> ambiguous, per the deck's own caveat: either the bath distorts
                 the state, or e affects mu directly and not only through Theta.
                 Agreement is the informative outcome; disagreement is not a
                 clean falsification.

Prediction on record (ANALYSIS_PROTOCOL Amendment 5a, before the data existed):
agreement within 2 sigma.

Usage:  python3 analyze_nothermo.py
"""
import glob
import os
import re

import numpy as np

import nfit

GD = 1.0e-3
RX_NT = (r'log\.(?P<kind>nt|ntk)_g(?P<g>[0-9.eE+-]+)_e(?P<e>[0-9.]+)'
         r'_mu(?P<mu>[0-9.]+)_s(?P<s>\d+)$')
RX_RE = r'log\.e(?P<e>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
ZMIN = 3.0


def athermal():
    out = {}
    for p in sorted(glob.glob('sweep_nothermo2/log.*')):
        m = re.match(RX_NT, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        g = m.groupdict()
        out.setdefault((g['kind'], float(g['g']), float(g['e']), float(g['mu'])),
                       []).append(q)
    return out


def thermostatted():
    cells = {}
    for p in glob.glob('sweep_restit/log.e*'):
        m = re.match(RX_RE, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        cells.setdefault((float(m.group('e')), float(m.group('mu'))), []).append(q)
    return cells


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def main():
    A, T = athermal(), thermostatted()
    if not A:
        print('sweep_nothermo2 is empty -- nothing to do yet.')
        return
    print('=' * 92)
    print('THE ATHERMAL CONTROL   --   does removing the bath land on the same mu(Theta)?')
    print('=' * 92)
    print(f'  {"e":>5}{"mu_g":>6}{"gdot":>10}{"n runs":>8}{"Theta_shear":>13}'
          f'{"/(gdot d)^2":>12}{"mu":>9}{"Z":>7}{"I":>10}')
    for k in sorted(A):
        kind, g, e, mg = k
        if kind != 'nt':
            continue
        r = A[k]
        th = float(np.mean([x['Theta'] for x in r]))
        print(f'  {e:>5g}{mg:>6g}{g:>10.3e}{len(r):>8}{th:>13.4e}'
              f'{th/g**2:>12.0f}{np.mean([x["mu"] for x in r]):>9.4f}'
              f'{np.mean([x.get("Z", np.nan) for x in r]):>7.2f}'
              f'{np.mean([x["I"] for x in r]):>10.3e}')

    print()
    print('  OVERLAY: the thermostatted curve (sweep_restit, same e, mu_g, gdot,')
    print('  packing) evaluated at the athermal Theta_shear.')
    print(f'  {"e":>5}{"mu_g":>6}{"Th_shear":>12}{"thermostatted Theta":>22}'
          f'{"extrapolation":>15}{"mu athermal":>13}{"mu predicted":>14}{"sigma":>8}')
    zs = []
    for k in sorted(A):
        kind, g, e, mg = k
        if kind != 'nt' or abs(g - GD) > 1e-12:
            continue                      # overlay only where gdot matches
        if (e, mg) not in T:
            continue
        runs = T[(e, mg)]
        ts = jammed(runs)
        if len(ts) < 4:
            print(f'  {e:>5g}{mg:>6g}   thermostatted cell VOID ({len(ts)} setpoints)')
            continue
        _, byT = nfit.surviving_setpoints(runs, z_min=ZMIN, restrict_to=ts)
        th_t = np.array([x['Theta'] for t in ts for x in byT[t] if x['Theta'] > 0])
        ath = A[k]
        th_a = float(np.mean([x['Theta'] for x in ath]))
        mu_a = float(np.mean([x['mu'] for x in ath]))
        sd_a = (float(np.std([np.log(x['mu']) for x in ath], ddof=1))
                / np.sqrt(len(ath)) if len(ath) > 1 else np.nan)
        r = nfit.fit_local_sys(runs, th_a, restrict_to=ts, z_min=ZMIN)
        if r is None:
            continue
        # value of the thermostatted quadratic AT Theta_shear, with its error
        sel = [x for t in ts for x in byT[t] if x['Theta'] > 0 and x['mu'] > 0]
        x = np.log([q['Theta'] for q in sel]) - np.log(th_a)
        y = np.log([q['mu'] for q in sel])
        M = np.vstack([np.ones_like(x), x, x ** 2]).T
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        res = y - M @ c
        cov = (float(res @ res) / max(len(x) - 3, 1)) * np.linalg.inv(M.T @ M)
        pred, epred = float(c[0]), float(np.sqrt(cov[0, 0]))
        # how far outside the fitted range is Theta_shear?
        if th_a < th_t.min():
            ex = f'{np.log(th_t.min()/th_a):+.2f} below'
        elif th_a > th_t.max():
            ex = f'{np.log(th_a/th_t.max()):+.2f} above'
        else:
            ex = 'interior'
        d = np.log(mu_a) - pred
        se = float(np.hypot(epred, sd_a if np.isfinite(sd_a) else 0.0))
        z = d / se if se > 0 else np.nan
        zs.append(z)
        print(f'  {e:>5g}{mg:>6g}{th_a:>12.3e}   [{th_t.min():.2e},{th_t.max():.2e}]'
              f'{ex:>15}{mu_a:>13.4f}{np.exp(pred):>14.4f}{z:>8.1f}')
    if zs:
        zs = np.array([z for z in zs if np.isfinite(z)])
        print(f'\n  RMS discrepancy {np.sqrt((zs**2).mean()):.1f} sigma over {len(zs)}'
              f' cells; max {np.abs(zs).max():.1f} sigma;'
              f' {int((np.abs(zs) > 2).sum())} above 2 sigma')
        print('  PASS (Amendment 5a) requires every cell within 2 sigma.')

    # the starting-kick control
    kk = [k for k in A if k[0] == 'ntk']
    if kk:
        print('\n  KICK CONTROL: the deck needs a non-zero temperature at step 0.')
        print('  The steady state must not remember it.')
        for k in kk:
            base = ('nt',) + k[1:]
            if base not in A:
                continue
            a = float(np.mean([x['Theta'] for x in A[base]]))
            b = float(np.mean([x['Theta'] for x in A[k]]))
            ma = float(np.mean([x['mu'] for x in A[base]]))
            mb = float(np.mean([x['mu'] for x in A[k]]))
            print(f'   e={k[2]:g} mu_g={k[3]:g}:  kick x1  Theta={a:.4e} mu={ma:.4f}'
                  f'   |  kick x10  Theta={b:.4e} mu={mb:.4f}'
                  f'   |  dTheta {abs(b/a-1)*100:.1f}%  dmu {abs(mb/ma-1)*100:.1f}%')


if __name__ == '__main__':
    main()
