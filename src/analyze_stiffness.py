"""
Phase 2 analysis: does n depend on grain stiffness at fixed k_t/k_n?

Reads sweep_stiffness/ (see run_stiffness.sh) and answers two questions.

1. THE TIMESCALE TEST. Scaling Emod with `tangential mindlin NULL` moves k_n
   and k_t together, so k_t/k_n is pinned while the Hertzian contact time
   moves as E^(-2/5) (a 100x span in E is a ~6.3x span in contact time). If n
   varies, contact DURATION matters and the known k_t and tdamp dependences
   are plausibly one effect. If n is flat, the k_t dependence is genuinely
   about tangential ELASTICITY.

2. AN ORTHOGONAL LEVER ON dZ_eff. Stiffer grains overlap less and coordinate
   less: Z falls from 4.04 at E=1e5 to 3.17 at E=1e6, at fixed pressure AND
   fixed friction. Every dZ_eff value measured so far was reached by varying
   mu_g, so the collapse test has never separated "n depends on dZ_eff" from
   "n and dZ_eff both depend on mu_g". This does.

   Caveat, stated because it limits what can be concluded here: dZ_eff needs
   chi, which needs the force-resolved tangential dump that this input does
   not write. So this script reports Z, not dZ_eff. Closing the loop needs a
   tracking sweep over Emod (in.contact_tracking_forces_2d).

THE TWO QUESTIONS ARE NOT INDEPENDENT -- read the verdict accordingly.
Changing Emod moves the contact time AND the coordination, because overlap
scales as (P/E)^(2/3). So "n varies with E" does NOT by itself establish a
timescale effect; it could be the coordination channel acting through Z. The
two separate only via the collapse: if n(E) lies on the same n(dZ_eff) curve
traced by varying mu_g, the effect is structural and there is no separate
timescale channel. If n(E) departs from that curve, the residual IS the
timescale. So the "VARIES/FLAT" verdict printed below is necessary but not
sufficient -- a FLAT result cleanly kills the timescale hypothesis, whereas a
VARIES result must be followed by the Emod tracking sweep before it means
anything.

Setpoint validity is enforced as elsewhere: a run whose I drifts more than 3%
from the cell median violates the fixed-I premise and is dropped.

Usage:  python3 analyze_stiffness.py [--dir sweep_stiffness]
"""
import argparse
import glob
import os
import re
import numpy as np

E0 = 1.0e5          # reference modulus
LABEL = re.compile(r'E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                   r'_T(?P<T>[0-9.eE+-]+)_s(?P<seed>\d+)$')
I_TOL = 0.03


def parse(path):
    """Average the thermo block after the measurement marker."""
    lines = open(path).readlines()
    start = next((i for i, l in enumerate(lines)
                  if 'BEGIN MEASUREMENT WINDOW' in l), None)
    if start is None:
        return None
    header, rows = None, []
    for l in lines[start:]:
        t = l.split()
        if t and t[0] == 'Step':
            header = t
            continue
        if header is None:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            if rows:
                break
            continue
        if len(v) == len(header):
            rows.append(v)
    if not rows:
        return None
    a = np.array(rows)[len(rows) // 3:]
    c = {h: a[:, i] for i, h in enumerate(header)}
    return dict(P=c['v_P'].mean(), mu=c['v_muI'].mean(),
                Theta=c['v_Theta'].mean(), I=c['v_Iiner'].mean(),
                Z=c['v_Zc'].mean() if 'v_Zc' in c else np.nan)


def fit(th, mu):
    th, mu = np.asarray(th, float), np.asarray(mu, float)
    m = np.isfinite(th) & np.isfinite(mu) & (th > 0) & (mu > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sweep_stiffness')
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, 'log.*'))):
        m = LABEL.match(os.path.basename(p)[4:])
        if not m:
            continue
        r = parse(p)
        if r is None:
            continue
        r.update(E=float(m.group('E')), mu_g=float(m.group('mu')),
                 Tgran=float(m.group('T')), seed=int(m.group('seed')))
        rows.append(r)
    if not rows:
        print('no runs parsed yet')
        return

    Es = sorted({r['E'] for r in rows})
    mus = sorted({r['mu_g'] for r in rows})

    print(f'{len(rows)} runs parsed\n')
    print('n(mu_g, Emod)   [contact time ~ E^-2/5, so 100x in E = 6.3x in t_c]')
    print(f'{"":10}' + ''.join(f'{e:>16.0e}' for e in Es))
    grid, err = {}, {}
    for mg in mus:
        line = f'mu_g={mg:<5g}'
        for E in Es:
            s = [r for r in rows if r['mu_g'] == mg and r['E'] == E]
            if len({r['Tgran'] for r in s}) < 3:
                line += f'{"-":>16}'
                continue
            Im = np.median([r['I'] for r in s])
            s = [r for r in s if abs(r['I'] / Im - 1) <= I_TOL]
            n, e = fit([r['Theta'] for r in s], [r['mu'] for r in s])
            grid[(mg, E)], err[(mg, E)] = n, e
            line += f'{n:+.3f}+/-{e:.3f}'.rjust(16)
        print(line)

    print('\nZ(mu_g, Emod)   [the orthogonal lever on coordination]')
    print(f'{"":10}' + ''.join(f'{e:>16.0e}' for e in Es))
    for mg in mus:
        line = f'mu_g={mg:<5g}'
        for E in Es:
            s = [r for r in rows if r['mu_g'] == mg and r['E'] == E]
            line += (f'{np.mean([r["Z"] for r in s]):>16.3f}' if s
                     else f'{"-":>16}')
        print(line)

    print('\n--- d n / d ln(E) at fixed k_t/k_n ---')
    for mg in mus:
        v = [(E, grid[(mg, E)]) for E in Es
             if np.isfinite(grid.get((mg, E), np.nan))]
        if len(v) < 3:
            continue
        x = np.log(np.array([a for a, _ in v]) / E0)
        y = np.array([b for _, b in v])
        A = np.vstack([x, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        s2 = np.sum(r ** 2) / max(len(x) - 2, 1)
        se = np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])
        sig = abs(c[0]) / se if se > 0 else np.nan
        verdict = 'FLAT (elasticity)' if sig < 2 else 'VARIES (timescale)'
        print(f'  mu_g={mg:<5g} slope={c[0]:+.4f}+/-{se:.4f} ({sig:4.1f} sigma)'
              f'  n spans {y.min():.3f}-{y.max():.3f}   -> {verdict}')


if __name__ == '__main__':
    main()
