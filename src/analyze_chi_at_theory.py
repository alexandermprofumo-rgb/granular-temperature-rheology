"""Two theory-motivated tests, on data already on disk. No simulation.

TEST 1 -- IS chi SET BY Z?
Generalised isostaticity (Shundyak/van Hecke, Somfai et al.) counts constraints
in 2D: a non-sliding contact supplies 2 (normal + tangential), a SLIDING one
supplies 1 (its tangential force is determined, f_t = mu_g f_n, so it is not an
unknown), against 3 degrees of freedom per grain. Isostatic when

    (Z/2) [ 2(1-chi) + chi ] = 3      ->      Z_iso(chi) = 6 / (2 - chi)

Checks: chi = 0 -> Z = 3 (frictional isostatic); chi = 1 -> Z = 6, MORE than
frictionless (4), because full mobilisation removes unknowns while torque
balance still supplies equations.

If chi is a single-valued function of Z across levers that move friction,
stiffness, pressure and thermostat coupling independently, then "what sets chi"
reduces to "what sets Z" -- and Z has a large literature. That is the useful
result whether or not the isostatic FORM is the right one.

TEST 2 -- IS a_t SET BY THE COULOMB BOUND?
Every tangential force obeys |f_t| <= mu_g f_n, and only the chi-fraction sits
at the bound, so dimensionally a_t ~ mu_g * chi * (structural factor). If a
single power law in (mu_g, chi, a_c) describes a_t across all four sweeps, then
C_t -- the one channel that failed against every candidate -- stops being an
orphan.

Usage:  python3 analyze_chi_at_theory.py
"""
import csv
import re

import numpy as np

NGRAIN = 4000
FILES = {'friction': 'chan_steady2d.csv', 'stiffness': 'chan_lev_E.csv',
         'pressure': 'chan_lev_P.csv', 'tangential': 'chan_lev_kt.csv'}


def gated_keys():
    """(sweep, cell, Tgran) triples that survive all four validity gates.

    The first pass pooled EVERY run, including cold setpoints that fail the
    barostat and jamming gates -- which is why Z ranged down to 0.58, an
    impossible value for a jammed 2D pack. A null drawn from ungated data is
    worthless, so gate first.
    """
    import glob, os
    import nfit
    keep = set()
    specs = [
        ('friction', 'sweep_steady2d/log.mu*',
         r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ()),
        ('stiffness', 'sweep_lev_E/log.E*',
         r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('E',)),
        ('pressure', 'sweep_lev_P/log.P*',
         r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('P',)),
        ('tangential', 'sweep_lev_kt/log.k*',
         r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('kt',)),
    ]
    for lever, pat, rx, extra in specs:
        cells = {}
        for f in glob.glob(pat):
            m = re.match(rx, os.path.basename(f))
            if not m: continue
            q = nfit.parse_log(f)
            if not q: continue
            q['Tgran'] = float(m.group('T'))
            key = tuple([float(m.group('mu'))] + [float(m.group(k)) for k in extra])
            cells.setdefault(key, []).append(q)
        for key, runs in cells.items():
            Ts, byT = nfit.surviving_setpoints(runs)
            for t in Ts:
                z = [x['Z'] for x in byT[t] if 'Z' in x]
                if z and np.mean(z) < 3.0:      # jamming gate
                    continue
                keep.add((lever, key, round(t, 12)))
    return keep


def load():
    keep = gated_keys()
    rows = []
    for lever, f in FILES.items():
        try:
            rd = list(csv.DictReader(open(f)))
        except OSError:
            continue
        for x in rd:
            r = {k: (float(v) if k != 'seed' else v) for k, v in x.items()}
            r['lever'] = lever
            r['Z'] = 2.0 * r['ncon'] / NGRAIN
            key = tuple([r['mu_g']] + [r[k] for k in ('E','P','kt') if k in r])
            if (lever, key, round(r['Tgran'], 12)) not in keep:
                continue
            rows.append(r)
    return rows


def wfit(A, y, w=None):
    w = np.ones(len(y)) if w is None else w
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    return c, cov, y - A @ c


def main():
    rows = load()
    print(f'{len(rows)} runs from {len(set(r["lever"] for r in rows))} levers: '
          + ', '.join(sorted(set(r['lever'] for r in rows))))
    fr = [r for r in rows if r['mu_g'] > 0 and r['chi'] > 0 and r['Z'] > 0]
    print(f'{len(fr)} with mu_g > 0 and chi > 0\n')

    # ---------------------------------------------------------------- TEST 1
    print('=' * 74)
    print('TEST 1   is chi a function of Z alone?')
    print('=' * 74)
    Z = np.array([r['Z'] for r in fr]); chi = np.array([r['chi'] for r in fr])
    iso = 2.0 - 6.0 / Z
    print(f'   Z range   {Z.min():.2f} to {Z.max():.2f}')
    print(f'   chi range {chi.min():.4f} to {chi.max():.4f}\n')
    print(f'   {"Z bin":>12}{"n":>6}{"chi measured":>16}{"chi = 2 - 6/Z":>16}')
    edges = np.percentile(Z, np.linspace(0, 100, 9))
    for a, b in zip(edges[:-1], edges[1:]):
        m = (Z >= a) & (Z < b)
        if m.sum() < 5:
            continue
        print(f'   {a:5.2f}-{b:5.2f}{m.sum():>6}{chi[m].mean():>11.4f} '
              f'+/-{chi[m].std():.4f}{iso[m].mean():>16.4f}')
    resid = chi - iso
    print(f'\n   isostatic form:  mean(chi - chi_iso) = {resid.mean():+.4f}, '
          f'rms {np.sqrt((resid**2).mean()):.4f}')
    print('   -> the isostatic FORM is not expected to hold exactly in flow '
          '(Z sits above\n      isostatic); what matters is whether chi is '
          'single-valued in Z at all.')

    # is chi single-valued in Z, regardless of form? fit a cubic in Z, look at
    # scatter and at whether the levers separate
    A = np.vstack([np.ones_like(Z), Z, Z ** 2, Z ** 3]).T
    c, _, res = wfit(A, chi)
    tot = chi - chi.mean()
    print(f'\n   cubic chi(Z) over ALL levers: residual rms {res.std():.4f} '
          f'against chi spread {chi.std():.4f}  (R^2 = {1-res.var()/tot.var():.3f})')
    print(f'   {"lever":>12}{"n":>6}{"mean residual":>16}{"in units of rms":>18}')
    off = {}
    for lv in sorted(set(r['lever'] for r in fr)):
        m = np.array([r['lever'] == lv for r in fr])
        off[lv] = res[m].mean()
        print(f'   {lv:>12}{m.sum():>6}{res[m].mean():>+16.4f}'
              f'{res[m].mean()/res.std():>18.2f}')
    spread = max(off.values()) - min(off.values())
    print(f'\n   lever-to-lever offset spread: {spread:.4f} '
          f'({spread/res.std():.1f} x the residual rms)')
    print('   -> chi IS a function of Z alone only if the levers do not '
          'separate.')

    # ---------------------------------------------------------------- TEST 2
    print()
    print('=' * 74)
    print('TEST 2   is a_t set by the Coulomb bound,  a_t ~ mu_g^a chi^b a_c^c ?')
    print('=' * 74)
    ok = [r for r in fr if r['a_t'] > 0 and r['a_c'] > 0]
    print(f'   {len(ok)} runs with a_t > 0')
    y = np.log([r['a_t'] for r in ok])
    lm = np.log([r['mu_g'] for r in ok])
    lc = np.log([r['chi'] for r in ok])
    la = np.log([r['a_c'] for r in ok])
    for name, cols in (('a_t ~ mu_g^a', [lm]),
                       ('a_t ~ mu_g^a chi^b', [lm, lc]),
                       ('a_t ~ mu_g^a chi^b a_c^c', [lm, lc, la])):
        A = np.vstack([np.ones(len(y))] + cols).T
        c, _, res = wfit(A, y)
        exps = '  '.join(f'{e:+.3f}' for e in c[1:])
        print(f'   {name:<28} exponents {exps:<26} '
              f'residual rms(ln a_t) {res.std():.4f}  '
              f'R^2 = {1-res.var()/np.var(y):.3f}')
    A = np.vstack([np.ones(len(y)), lm, lc, la]).T
    c, _, res = wfit(A, y)
    print(f'\n   full form, residuals by lever:')
    for lv in sorted(set(r['lever'] for r in ok)):
        m = np.array([r['lever'] == lv for r in ok])
        print(f'   {lv:>12}{m.sum():>6}  mean residual {res[m].mean():+.4f}'
              f'  ({res[m].mean()/res.std():+.2f} rms)')
    print(f'\n   a residual rms of ~0.1 in ln a_t is ~10% in a_t.')


if __name__ == '__main__':
    main()
