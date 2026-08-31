"""The 3D three-channel decomposition, EXACT -- no closure, no fitted prefactor.

Section 6 could not do this. Its 3D channels were built by calibrating a
prefactor k on the frictionless cells and then closing A_t = mu/k - A_c - A_n at
every other friction, so A_t absorbed every error in k, A_c and A_n and carried
a 0.022 noise floor. The steady decks dump the tangential force on the
production runs themselves, so the decomposition can be written down exactly.

    S    = sum_c (f_n + f_t) (x) d              total contact stress
    P    = tr(S)/D                              (tr of the tangential part is
                                                 zero: f_t . n = 0)
    s^   = dev(S)/|dev(S)|                      unit deviator of the TOTAL stress
    v_c  = l_c / sum l                          branch weights
    w_c  = l_c f_n^c / sum(l f_n)               branch-and-force weights
    F    = sum v_c n(x)n,   G = sum w_c n(x)n

Since the normal stress is exactly sum_c l_c f_n^c n(x)n = tr(S) . G,

    mu_n = D . dev(G):s^      and      mu = D.A_c + D.A_n + mu_t   EXACTLY

with A_c = dev(F):s^ (geometric) and A_n = dev(G-F):s^ (normal-force). No
prefactor is fitted and nothing is closed; the identity is checked to machine
precision below. Differentiating gives n = C_c + C_n + C_t as before.

Also runs the master-curve collapse test on each 3D channel and on the exact
split mu_n / mu_t in both dimensions.

Usage:  python3 analyze_channels_3d_steady.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit
from analyze_rb import local
from analyze_collapse_channels import collapse

CACHE = 'steady3d_channels.csv'
RX = re.compile(r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
NCOL, D = 14, 3


def frames(path, keep_last=8):
    out = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                k = int(f.readline())
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                rows = []
                for _ in range(k):
                    t = f.readline().split()
                    if len(t) >= NCOL:
                        rows.append([float(x) for x in t[:NCOL]])
                if rows:
                    out.append(np.array(rows))
    return out[-keep_last:]


def channels(path):
    acc = []
    for a in frames(path):
        d, fnv, ftv = a[:, 4:7], a[:, 7:10], a[:, 10:13]
        L = np.linalg.norm(d, axis=1)
        live = (np.linalg.norm(fnv, axis=1) > 0) & (L > 0)
        if live.sum() < 200:
            continue
        d, fnv, ftv, L = d[live], fnv[live], ftv[live], L[live]
        nh = d / L[:, None]
        Sn = np.einsum('ci,cj->ij', fnv, d)
        St = np.einsum('ci,cj->ij', ftv, d)
        Sn = .5 * (Sn + Sn.T); St = .5 * (St + St.T)
        S = Sn + St
        tr = np.trace(S)
        if tr <= 0:
            continue
        P = tr / D
        dev = S - np.eye(D) * tr / D
        nrm = np.sqrt(np.sum(dev * dev))
        if nrm <= 0:
            continue
        sh = dev / nrm
        mu = float(np.sum(dev * sh) / P)
        mu_n = float(np.sum((Sn - np.eye(D) * np.trace(Sn) / D) * sh) / P)
        mu_t = float(np.sum((St - np.eye(D) * np.trace(St) / D) * sh) / P)
        fs = np.einsum('ij,ij->i', fnv, nh)
        v = L / L.sum()
        wt = L * fs
        w = wt / wt.sum()
        F = np.einsum('c,ci,cj->ij', v, nh, nh)
        G = np.einsum('c,ci,cj->ij', w, nh, nh)
        Fd = F - np.eye(D) * np.trace(F) / D
        Gd = G - np.eye(D) * np.trace(G) / D
        A_c = float(np.sum(Fd * sh))
        A_n = float(np.sum((Gd - Fd) * sh))
        acc.append((mu, mu_n, mu_t, A_c, A_n,
                    (D * A_c + D * A_n + mu_t - mu) / max(abs(mu), 1e-30)))
    return np.mean(acc, axis=0) if acc else None


def build(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('mu_g', 'Tgran', 'Theta', 'mu', 'mu_n', 'mu_t', 'A_c', 'A_n'):
                r[k] = float(r[k])
        return rows, None
    rows, worst = [], 0.0
    paths = sorted(glob.glob('sweep_steady3d/dump.contacts.mu*'))
    for i, p in enumerate(paths):
        m = RX.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(f'sweep_steady3d/log.mu{m.group("mu")}'
                           f'_T{m.group("T")}_s{m.group("s")}')
        if not q:
            continue
        g = channels(p)
        if g is None:
            continue
        worst = max(worst, abs(g[5]))
        rows.append(dict(mu_g=float(m.group('mu')), Tgran=float(m.group('T')),
                         seed=m.group('s'), Theta=q['Theta'], mu=g[0],
                         mu_n=g[1], mu_t=g[2], A_c=g[3], A_n=g[4]))
        if (i + 1) % 40 == 0:
            print(f'   ... {i+1}/{len(paths)}', file=sys.stderr)
    with open(CACHE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows, worst


def main():
    rows, worst = build('--recache' in sys.argv)
    print(f'{len(rows)} runs from sweep_steady3d (strain 1.0, a_t measured)')
    if worst is not None:
        # The tolerance is set by the DUMP, not by float64. LAMMPS writes ~6
        # significant figures, so f_t is only perpendicular to n to ~4e-7
        # (median |f_t.n|/|f_t| = 3.8e-7, max 3.7e-6) and tr(S_t) is not
        # exactly zero. The identity therefore cannot hold better than ~1e-7.
        print(f'exactness of  mu = 3A_c + 3A_n + mu_t : max relative error '
              f'{worst:.2e}'
              f'   {"(dump write precision -- exact)" if worst < 1e-6 else "<-- COLUMN MAPPING WRONG"}')

    RXL = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
    cm = {}
    for k, v in nfit.load_sweep('sweep_steady3d/log.mu*', RXL).items():
        cm.setdefault(float(dict(k)['mu']), []).extend(v)
    mus = sorted(set(r['mu_g'] for r in rows) & set(cm))
    T0, _ = nfit.common_theta0([cm[m] for m in mus])
    print(f'\nTheta_0 = {T0:.3e}\n')
    print('=' * 78)
    print('3D CHANNELS, EXACT   n = C_c + C_n + C_t')
    print('=' * 78)
    print(f'{"mu_g":>6}{"C_c":>9}{"C_n":>9}{"C_t":>9}{"sum":>9}{"n fitted":>18}'
          f'{"  shares c / n / t":>22}')
    Dd, Ee = [], []
    for mg in mus:
        Ts, _ = nfit.surviving_setpoints(cm[mg])
        keep = {round(t, 12) for t in Ts}
        sel = [r for r in rows if r['mu_g'] == mg
               and round(r['Tgran'], 12) in keep and r['Theta'] > 0]
        if len(sel) < 8:
            continue
        x = np.log([r['Theta'] for r in sel]); x0 = np.log(T0)
        gm = local(x, [r['mu'] for r in sel], x0)
        if gm is None or gm[0] <= 0:
            continue
        C = {'c': -D * local(x, [r['A_c'] for r in sel], x0)[1] / gm[0],
             'n': -D * local(x, [r['A_n'] for r in sel], x0)[1] / gm[0],
             't': -local(x, [r['mu_t'] for r in sel], x0)[1] / gm[0]}
        tot = sum(C.values())
        f = nfit.fit_local_sys(cm[mg], T0)
        if not f:
            continue
        Dd.append(tot - f['n']); Ee.append(f['tot'])
        sh = ' / '.join(f'{C[k]/tot:5.2f}' for k in 'cnt') if abs(tot) > 1e-9 else ''
        print(f'{mg:>6g}{C["c"]:>9.4f}{C["n"]:>9.4f}{C["t"]:>9.4f}{tot:>9.4f}'
              f'{f["n"]:>12.4f}+/-{f["tot"]:<5.3f}   {sh}')
    Dd, Ee = np.array(Dd), np.array(Ee)
    rms = np.sqrt(((Dd / Ee) ** 2).mean())
    print(f'\nGATE  sum C_i vs fitted n: mean {Dd.mean():+.4f}, '
          f'RMS {np.sqrt((Dd**2).mean()):.4f}, {rms:.1f} sigma -> '
          f'{"USABLE" if rms < 2 else "FAILS"}')

    # collapse tests
    print()
    print('=' * 78)
    print('COLLAPSE TESTS on the 3D channels and on the exact split')
    print('=' * 78)

    def cells(src, key, cmap, mumin=0.0):
        out = []
        for mg in sorted(cmap):
            if mg < mumin:
                continue
            Ts, _ = nfit.surviving_setpoints(cmap[mg])
            keep = {round(t, 12) for t in Ts}
            sel = [r for r in src if r['mu_g'] == mg
                   and round(r['Tgran'], 12) in keep and r[key] > 0]
            if len(sel) >= 8:
                out.append((mg, np.log([r['Theta'] for r in sel]),
                            np.log([r[key] for r in sel])))
        return out

    for key, mumin, lab in (('A_c', 0.0, '3D A_c geometric'),
                            ('A_n', 0.0, '3D A_n normal-force'),
                            ('mu_n', 0.0, '3D mu_n (exact split)'),
                            ('mu_t', 0.05, '3D mu_t (exact split)')):
        c = cells(rows, key, cm, mumin)
        if len(c) >= 4:
            collapse(c, lab)

    r2 = list(csv.DictReader(open('steady2d_channels.csv')))
    for r in r2:
        for k in ('mu_g', 'Tgran', 'Theta', 'mu', 'a_c', 'a_n', 'a_t'):
            r[k] = float(r[k])
    cm2 = {}
    for k, v in nfit.load_sweep('sweep_steady2d/log.mu*', RXL).items():
        cm2.setdefault(float(dict(k)['mu']), []).extend(v)
    # 2D mu_n = (a_c + a_n)/2 * 2 = a_c + a_n ; mu_t = a_t  (up to the same
    # normalisation, which cancels in a log-slope)
    for r in r2:
        r['mu_n'] = r['a_c'] + r['a_n']
        r['mu_t'] = r['a_t']
    for key, mumin, lab in (('mu_n', 0.0, '2D mu_n (exact split)'),
                            ('mu_t', 0.05, '2D mu_t (exact split)')):
        c = cells(r2, key, cm2, mumin)
        if len(c) >= 4:
            collapse(c, lab)


if __name__ == '__main__':
    main()
