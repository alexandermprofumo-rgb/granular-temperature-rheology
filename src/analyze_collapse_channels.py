"""Does any CHANNEL collapse, even though mu(Theta) does not?

mu(Theta; mu_g) admits no master-curve collapse: the curves change shape with
friction (curvature runs -0.032 to +0.093), and a shift-collapse is rejected at
F = 20.8, p ~ 0. But mu = (a_c + a_n + a_t)/2 exactly, so mu failing to collapse
does not mean its parts fail -- a sum of three collapsing functions with
different Theta* need not itself collapse.

If a channel does collapse, that is the first positive organising law in the
project, and it localises where the friction dependence actually lives.

Same three nested models as for mu, per channel:
    M0  independent quadratic per cell           3 x ncell
    M1  master curve + vertical AND horizontal shift   4 + 2 x ncell
    M2  master curve + vertical shift only             4 + 1 x ncell
plus M1s: one SHARED Theta*(mu_g) across all three channels, which is the
strongest hypothesis -- a single characteristic temperature organising the
whole decomposition.

Pure analysis: uses sweep_steady2d dumps, no new simulations.

Usage:  python3 analyze_collapse_channels.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy import optimize, stats

import nfit
from analyze_rb import frames

CACHE = 'steady2d_channels.csv'
RX = re.compile(r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')


def rb_of(path):
    """(mu, a_c, a_n, a_t), branch-length weighted so the identity is exact."""
    acc = []
    for a in frames(path):
        live = np.linalg.norm(a[:, 6:8], axis=1) > 0
        if live.sum() < 100:
            continue
        b = a[live]
        d = b[:, 4:6]
        L = np.linalg.norm(d, axis=1)
        dn = d / L[:, None]
        fnv, ftv = b[:, 6:8], b[:, 8:10]
        S = np.einsum('ci,cj->ij', fnv + ftv, d)
        S = 0.5 * (S + S.T)
        P = (S[0, 0] + S[1, 1]) / 2.0
        if P <= 0:
            continue
        dev = complex((S[0, 0] - S[1, 1]) / 2.0, S[0, 1])
        ths = 0.5 * np.angle(dev)
        th = np.arctan2(dn[:, 1], dn[:, 0]) % np.pi
        that = np.stack([-dn[:, 1], dn[:, 0]], axis=1)
        fs = np.einsum('ij,ij->i', fnv, dn)
        ts = np.einsum('ij,ij->i', ftv, that)
        w = L / L.sum()
        f0 = float(np.sum(w * fs))
        ph = np.exp(-2j * ths)
        ac = 2 * (np.sum(w * np.exp(2j * th)) * ph).real
        acn = 2 * (np.sum(w * fs * np.exp(2j * th)) / f0 * ph).real
        at = -2 * (np.sum(w * ts * np.exp(2j * th)) / f0 * ph).imag
        acc.append((abs(dev) / P, ac, acn - ac, at))
    return np.mean(acc, axis=0) if acc else None


def build(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('mu_g', 'Tgran', 'Theta', 'mu', 'a_c', 'a_n', 'a_t'):
                r[k] = float(r[k])
        return rows
    rows = []
    paths = sorted(glob.glob('sweep_steady2d/dump.contacts.mu*'))
    for i, p in enumerate(paths):
        m = RX.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(f'sweep_steady2d/log.mu{m.group("mu")}'
                           f'_T{m.group("T")}_s{m.group("s")}')
        if not q:
            continue
        g = rb_of(p)
        if g is None:
            continue
        rows.append(dict(mu_g=float(m.group('mu')), Tgran=float(m.group('T')),
                         seed=m.group('s'), Theta=q['Theta'],
                         mu=g[0], a_c=g[1], a_n=g[2], a_t=g[3]))
        if (i + 1) % 40 == 0:
            print(f'   ... {i+1}/{len(paths)}', file=sys.stderr)
    with open(CACHE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def master(u, g):
    return g[0] + g[1] * u + g[2] * u ** 2 + g[3] * u ** 3


def collapse(cells, label):
    """cells: list of (mu_g, x=lnTheta, y=lnX). Returns fitted shifts."""
    N = sum(len(x) for _, x, _ in cells)
    nc = len(cells)
    r0 = sum(((y - np.polyval(np.polyfit(x, y, 2), x)) ** 2).sum()
             for _, x, y in cells)
    k0 = 3 * nc

    def obj1(p):
        g, sh, vt = p[:4], p[4:4 + nc], p[4 + nc:]
        return sum(((y - vt[i] - master(x - sh[i], g)) ** 2).sum()
                   for i, (_, x, y) in enumerate(cells))

    def obj2(p):
        g, vt = p[:4], p[4:]
        return sum(((y - vt[i] - master(x - cells[i][1].mean(), g)) ** 2).sum()
                   for i, (_, x, y) in enumerate(cells))

    x0 = np.concatenate([[0, -0.1, 0, 0], [x.mean() for _, x, _ in cells],
                         [y.mean() for _, _, y in cells]])
    r = optimize.minimize(obj1, x0, method='Nelder-Mead',
                          options=dict(maxiter=300000, maxfev=300000,
                                       fatol=1e-14, xatol=1e-11))
    r = optimize.minimize(obj1, r.x, method='Powell',
                          options=dict(maxiter=300000, maxfev=300000))
    x2 = np.concatenate([[0, -0.1, 0, 0], [y.mean() for _, _, y in cells]])
    r2 = optimize.minimize(obj2, x2, method='Powell',
                           options=dict(maxiter=300000, maxfev=300000))
    k1, k2 = 4 + 2 * nc, 4 + nc
    F = ((r.fun - r0) / (k0 - k1)) / (r0 / (N - k0))
    p = 1 - stats.f.cdf(F, k0 - k1, N - k0)
    curv = [-2 * np.polyfit(x - x.mean(), y, 2)[0] for _, x, y in cells]
    print(f'\n--- {label} ---')
    print(f'   curvature per cell: ' +
          ' '.join(f'{c:+.3f}' for c in curv) +
          f'   (sd {np.std(curv):.3f})')
    print(f'   M0 independent  {k0:>3} params  rms {np.sqrt(r0/N):.4f}')
    print(f'   M1 shift        {k1:>3} params  rms {np.sqrt(r.fun/N):.4f}'
          f'   ({np.sqrt(r.fun/r0):.2f}x M0)')
    print(f'   M2 vertical     {k2:>3} params  rms {np.sqrt(r2.fun/N):.4f}')
    print(f'   M1 vs M0: F = {F:.2f}, p = {p:.2e}   -> '
          f'{"COLLAPSES" if p > 0.05 else ("near-collapse" if p > 1e-3 else "no collapse")}')
    sh = r.x[4:4 + nc]
    print('   Theta*: ' + '  '.join(f'{mg:g}:{np.exp(s):.2e}'
                                    for (mg, _, _), s in zip(cells, sh)))
    return np.array([c[0] for c in cells]), np.exp(sh), p


def main():
    rows = build('--recache' in sys.argv)
    print(f'{len(rows)} runs from sweep_steady2d (strain 1.0, a_t measured)')

    RXL = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
    cm = {}
    for k, v in nfit.load_sweep('sweep_steady2d/log.mu*', RXL).items():
        cm.setdefault(float(dict(k)['mu']), []).extend(v)

    def build_cells(key, mumin):
        out = []
        for mg in sorted(cm):
            if mg < mumin:
                continue
            Ts, _ = nfit.surviving_setpoints(cm[mg])
            keep = {round(t, 12) for t in Ts}
            sel = [r for r in rows if r['mu_g'] == mg
                   and round(r['Tgran'], 12) in keep and r[key] > 0]
            if len(sel) < 8:
                continue
            out.append((mg, np.log([r['Theta'] for r in sel]),
                        np.log([r[key] for r in sel])))
        return out

    res = {}
    for key, mumin, lab in (('mu', 0.0, 'mu  (all frictions)'),
                            ('mu', 0.12, 'mu  (frictional branch only)'),
                            ('a_c', 0.0, 'a_c  geometric fabric'),
                            ('a_n', 0.0, 'a_n  normal-force anisotropy'),
                            ('a_t', 0.05, 'a_t  tangential (mu_g > 0)')):
        cells = build_cells(key, mumin)
        if len(cells) >= 4:
            res[lab] = collapse(cells, lab)

    print('\n' + '=' * 70)
    print('DO THE CHANNELS SHARE ONE Theta*(mu_g)?')
    print('=' * 70)
    common = None
    for lab, (mg, ts, p) in res.items():
        if lab.startswith('a_'):
            m = np.log(mg[mg > 0]); t = np.log(ts[mg > 0])
            s, b = np.polyfit(m, t, 1)
            print(f'   {lab:<32} Theta* ~ mu_g^{s:+.2f}')
            common = (mg, ts) if common is None else common


if __name__ == '__main__':
    main()
