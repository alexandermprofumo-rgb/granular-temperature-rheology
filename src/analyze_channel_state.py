"""Are the CHANNELS state functions, even though n is not?

Everyone has tested whether n is a state function of some microstructural
variable X. Ten candidates, all fail. But n = C_c + C_n + C_t exactly, so n
failing does not imply its parts fail: a sum of three quantities each obeying
its own law need not itself obey one.

The sharpest untested pairing is C_t against chi. chi is the fraction of
contacts sitting at their Coulomb limit; C_t IS the tangential channel. Testing
a whole-system exponent against a tangential observable was always a mismatch.

Channel quantities. C_i = -(1/2mu) d(a_i)/dlnTheta carries mu in the
denominator, so it is not a clean per-channel object. For the two channels that
do not vanish we use the pure logarithmic derivative
    n_i = -dln(a_i)/dlnTheta
and keep the C form only for the tangential channel, where a_t -> 0 at low
friction makes ln(a_t) unusable.

TEST (as pre-registered in ANALYSIS_PROTOCOL.md section 5.4): pool the friction
lever and the stiffness lever, fit ONE curve Y(X), and require BOTH
    (a) chi2/dof ~ 1 for the single curve, and
    (b) no residual trend against the lever variable at >= 2 sigma.
Both conditions, exactly as chi was judged.

Usage:  python3 analyze_channel_state.py
"""
import csv
import glob
import json
import os
import re

import numpy as np

import nfit

NGRAIN = 4000
ZMIN = 3.0
CHANNELS = ('n_c', 'n_n', 'C_t', 'n_tot')
CANDS = ('Z', 'chi', 'cv_f', 'pr_f', 'f_mean', 'a_c', 'a_n', 'a_t')


def load(path):
    rows = []
    for x in csv.DictReader(open(path)):
        r = {k: (float(v) if k != 'seed' else str(int(float(v))))
             for k, v in x.items()}
        r['Z'] = 2.0 * r['ncon'] / NGRAIN
        rows.append(r)
    return rows


def thetas(sweep, rx):
    out = {}
    for p in glob.glob(f'{sweep}/log.*'):
        m = re.match(rx, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if q:
            g = m.groupdict()
            key = tuple([float(g['mu']), float(g['T']), g['s']]
                        + [float(g[k]) for k in ('E',) if k in g])
            out[key] = (q['Theta'], q.get('Z', np.nan))
    return out


def loc(x, y, x0, deg=2):
    """value and d/dx at x0 from a local polynomial fit."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < deg + 2:
        return None
    A = np.vander(x - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    cov = (float(r @ r) / max(len(x) - deg - 1, 1)) * np.linalg.inv(A.T @ A)
    return float(c[0]), float(c[1]), float(np.sqrt(cov[1, 1]))


def gated_setpoints(sweep, rx, keyfn, zmin=ZMIN):
    """Per cell, the setpoints surviving ALL FOUR protocol gates.

    ANALYSIS_PROTOCOL section 2 requires the barostat, fixed-I, thermostat and
    jamming gates, applied per SETPOINT, in that order. This script previously
    applied none of the first three and applied the jamming gate on the
    cell-MEAN Z, so its per-cell fits included setpoints where the thermostat
    was not in control and where I deviated by up to 35%. Those points
    destabilise the quadratic badly -- ungated, C_c at mu_g = 0.12 comes out
    -0.25 against +0.063 gated.
    """
    out = {}
    for k, runs in nfit.load_sweep(f'{sweep}/log.*', rx).items():
        Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
        keep = set()
        for t in Ts:
            z = [x['Z'] for x in byT[t] if 'Z' in x]
            if not z or np.mean(z) >= zmin:
                keep.add(round(t, 12))
        out[keyfn(dict(k))] = keep
    return out


def cells_from(rows, th, keyfn, T0, gates=None):
    """-> {cellkey: dict of channel values and candidate values at T0}"""
    grp = {}
    for r in rows:
        k = keyfn(r)
        if gates is not None and round(r['Tgran'], 12) not in gates.get(k, ()):
            continue
        tk = tuple([r['mu_g'], r['Tgran'], r['seed']]
                   + ([r['E']] if 'E' in r else []))
        if tk not in th:
            continue
        r = dict(r); r['Theta'] = th[tk][0]
        if r['Theta'] > 0:
            grp.setdefault(k, []).append(r)
    out = {}
    for k, rs in grp.items():
        if len(rs) < 8:
            continue
        x = np.log([r['Theta'] for r in rs]); x0 = np.log(T0)
        gm = loc(x, [r['mu'] for r in rs], x0)
        if gm is None or gm[0] <= 0:
            continue
        d = {}
        ok = True
        for nm, key in (('n_c', 'a_c'), ('n_n', 'a_n')):
            v = np.array([r[key] for r in rs])
            if (v <= 0).any():
                ok = False; break
            g = loc(x, np.log(v), x0)
            d[nm] = -g[1]; d[nm + '_e'] = g[2]
        if not ok:
            continue
        g = loc(x, [r['a_t'] for r in rs], x0)
        d['C_t'] = -g[1] / (2 * gm[0]); d['C_t_e'] = g[2] / (2 * gm[0])
        g = loc(x, np.log([r['mu'] for r in rs]), x0)
        d['n_tot'] = -g[1]; d['n_tot_e'] = g[2]
        for c in CANDS:
            d[c] = float(np.mean([r[c] for r in rs]))
        d['Zbar'] = float(np.mean([r['Z'] for r in rs]))
        out[k] = d
    return out


def main():
    RXF = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
    RXE = (r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
           r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    fr = load('chan_steady2d.csv')
    st = load('chan_lev_E.csv')
    thf = thetas('sweep_steady2d', RXF)
    the = thetas('sweep_lev_E', RXE)

    allth = [v[0] for v in list(thf.values()) + list(the.values()) if v[0] > 0]
    T0 = float(np.exp(np.mean(np.log(allth))))
    print(f'common Theta_0 = {T0:.3e}\n')

    gf = gated_setpoints('sweep_steady2d', RXF, lambda d: ('fric', float(d['mu'])))
    ge = gated_setpoints('sweep_lev_E', RXE,
                         lambda d: ('stiff', float(d['E']), float(d['mu'])))
    A = cells_from(fr, thf, lambda r: ('fric', r['mu_g']), T0, gf)
    B = cells_from(st, the, lambda r: ('stiff', r['E'], r['mu_g']), T0, ge)
    A = {k: v for k, v in A.items() if v['Zbar'] >= ZMIN}
    B = {k: v for k, v in B.items() if v['Zbar'] >= ZMIN}
    print(f'friction lever: {len(A)} cells;  stiffness lever: {len(B)} cells')
    print(f'(all four protocol gates, per setpoint; jamming at Z >= {ZMIN})\n')

    print(f'{"channel":<8}{"candidate":<10}{"chi2/dof":>10}{"trend vs lnE":>15}'
          f'{"  verdict":<10}')
    print('  (the trend sigma is inflated by sqrt(chi2/dof) where that exceeds 1,')
    print('   as ANALYSIS_PROTOCOL section 4 requires: a residual trend measured')
    print('   against errors the fit itself has already overshot is not a sigma.)')
    print('-' * 56)
    nres = 0
    RESULT = {}
    for ch in CHANNELS:
        for cand in CANDS:
            pts = []
            for src, d in (('f', A), ('s', B)):
                for k, v in d.items():
                    lev = 1.0e5 if src == 'f' else k[1]
                    if v[cand] > 0 and np.isfinite(v[ch]) and v[ch + '_e'] > 0:
                        pts.append((v[cand], v[ch], v[ch + '_e'], lev))
            if len(pts) < 8:
                continue
            x = np.log([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            e = np.array([p[2] for p in pts]); lE = np.log([p[3] for p in pts])
            if lE.std() == 0:
                continue
            M = np.vstack([np.ones_like(x), x, x ** 2]).T
            W = np.diag(1 / e ** 2)
            c = np.linalg.inv(M.T @ W @ M) @ M.T @ W @ y
            res = y - M @ c
            chi2 = float(((res / e) ** 2).sum()) / max(len(x) - 3, 1)
            Ml = np.vstack([np.ones_like(lE), lE]).T
            cov = np.linalg.inv(Ml.T @ W @ Ml)
            cl = cov @ Ml.T @ W @ res
            tz = (abs(cl[1]) / np.sqrt(cov[1, 1])) / np.sqrt(max(chi2, 1.0))
            nres += 1
            ok = chi2 < 2.0 and tz < 2.0
            RESULT[f'{ch}|{cand}'] = [chi2, tz, bool(ok)]
            print(f'{ch:<8}{cand:<10}{chi2:>10.1f}{tz:>13.1f}s  '
                  f'{"** PASSES **" if ok else "fails"}')
    npass = sum(1 for v in RESULT.values() if v[2])
    print(f'\n{nres} (channel, candidate) pairs tested -- multiplicity: at 2 sigma')
    print(f'you would expect ~{0.05*nres:.0f} false passes by chance.')
    print(f'{npass} passed.')
    # Cache, so that no other script and no report table ever carries a
    # hand-copied 2D number again. The earlier "1.1 / 0.8" was transcribed by
    # hand and never recomputed after the 2026-08-20 re-extraction; it is not
    # reproducible
    # under any frame policy, weighting, gate or Theta_0.
    json.dump({'Theta0': T0, 'n_fric_cells': len(A), 'n_stiff_cells': len(B),
               'npairs': nres, 'npass': npass, 'pairs': RESULT},
              open('channel_state_2d.json', 'w'), indent=1)
    print('cached -> channel_state_2d.json')


if __name__ == '__main__':
    main()
