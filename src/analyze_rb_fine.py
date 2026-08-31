"""
The channel decomposition at fine friction resolution (2D, Pconf = 10).

sweep_pichi resolves the channels at six frictions. sweep_matched2d has 496
contact dumps at fifteen, from mu_g = 0 to 1.0, spanning the peak and both
flanks -- which is what is needed to see how the channels produce the shape of
n(mu_g) rather than just its value at a handful of points.

ONE COMPROMISE, AND HOW IT IS CONTROLLED. These dumps were written by
in.granular_2d with `compute pl all pair/local dx dy dz fx fy`, so they carry
the branch vector and the NORMAL force only -- pair/local's force is purely
central. a_c and a_n are therefore measured, but a_t is not available. It is
recovered by closure from the identity validated in §7b:

    a_t = 2 mu - a_c - a_n        (mu from the thermo log)

so a_t absorbs the identity's ~1.5% residual as well as its own signal. That is
acceptable because a_t is the smallest channel (1-25% of n), but it means a_t
here is inferred, not measured, and must not be quoted as an independent check
of the identity.

THE FREE VALIDATION. At mu_g = 0 there is no tangential force at all, so a_t
must be exactly zero. Closure has to reproduce that from a_c, a_n and mu which
are measured independently of it. If a_t(mu_g=0) comes out near zero, the
procedure is sound; if not, either the identity or the extraction is wrong at
this pressure and nothing below should be believed.

Usage:  python3 analyze_rb_fine.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit
from analyze_rb import local

CACHE = 'rb_fine.csv'
NFRAMES = 10          # last N frames per dump; chi-like averages converge fast


def frames(path, keep_last=NFRAMES):
    """Frame arrays (index, dx, dy, fx, fy) -- the in.granular_2d layout."""
    blocks = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                n = int(f.readline())
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                rows = []
                for _ in range(n):
                    t = f.readline().split()
                    if len(t) >= 5:
                        rows.append([float(x) for x in t[:5]])
                if rows:
                    blocks.append(np.array(rows))
    return blocks[-keep_last:]


def anis(path, K=36):
    """(a_c, a_n) from branch vectors and normal forces."""
    acc = []
    for a in frames(path):
        d = a[:, 1:3]
        nrm = np.linalg.norm(d, axis=1)
        f = a[:, 3:5]
        fm = np.linalg.norm(f, axis=1)
        live = (fm > 0) & (nrm > 0)
        if live.sum() < 100:
            continue
        d = d[live]; nrm = nrm[live]; f = f[live]
        dn = d / nrm[:, None]
        fs = np.einsum('ij,ij->i', f, dn)          # signed normal magnitude
        # stress deviator direction, from the normal part only; the tangential
        # contribution to the principal ANGLE is small (the assembly is coaxial
        # to 1-2 deg, §7b) even though its contribution to tau is not
        S = np.einsum('ci,cj->ij', f, d)
        S = 0.5 * (S + S.T)
        ths = 0.5 * np.angle(complex((S[0, 0] - S[1, 1]) / 2, S[0, 1]))
        th = np.arctan2(dn[:, 1], dn[:, 0]) % np.pi
        idx = np.floor(th / np.pi * K).astype(int) % K
        cnt = np.bincount(idx, minlength=K).astype(float)
        if (cnt == 0).any():
            continue
        E = cnt / cnt.sum()
        fnb = np.array([fs[idx == k].mean() for k in range(K)])
        tc = (np.arange(K) + 0.5) * np.pi / K
        f0 = np.average(fnb, weights=cnt)
        ph = np.exp(-2j * ths)
        ac = (2 * np.sum(E * np.exp(2j * tc)) * ph).real
        acn = (2 * np.sum(E * fnb * np.exp(2j * tc)) / f0 * ph).real
        acc.append((ac, acn - ac))
    if not acc:
        return None
    return np.mean(acc, axis=0)


def build(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('mu_g', 'Tgran', 'mu', 'a_c', 'a_n', 'a_t', 'Theta'):
                r[k] = float(r[k])
        return rows
    rx = re.compile(r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    paths = sorted(glob.glob('sweep_matched2d/dump.contacts.mu*'))
    rows = []
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(f'sweep_matched2d/log.mu{m.group("mu")}'
                           f'_T{m.group("T")}_s{m.group("s")}')
        if not q or q['mu'] <= 0:
            continue
        got = anis(p)
        if got is None:
            continue
        ac, an = got
        rows.append(dict(mu_g=float(m.group('mu')), Tgran=float(m.group('T')),
                         seed=m.group('s'), mu=q['mu'], Theta=q['Theta'],
                         a_c=ac, a_n=an, a_t=2 * q['mu'] - ac - an))
        if (i + 1) % 60 == 0:
            print(f'   ... {i+1}/{len(paths)}', file=sys.stderr)
    if rows:
        with open(CACHE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows


def main():
    rows = build('--recache' in sys.argv)
    if not rows:
        print('no dumps parsed')
        return
    print(f'{len(rows)} runs\n')

    print('=' * 74)
    print('VALIDATION -- a_t recovered by closure must vanish at mu_g = 0')
    print('=' * 74)
    z = [r['a_t'] for r in rows if r['mu_g'] == 0]
    nz = [r['a_t'] for r in rows if r['mu_g'] >= 0.3]
    if z:
        print(f'   a_t(mu_g=0)   = {np.mean(z):+.4f} +/- {np.std(z):.4f}  ({len(z)} runs)')
    if nz:
        print(f'   a_t(mu_g>=0.3) = {np.mean(nz):+.4f} +/- {np.std(nz):.4f}  ({len(nz)} runs)')
    if z and nz and abs(np.mean(z)) < 3 * np.std(z) + 0.01:
        print('   -> consistent with zero at zero friction; closure is sound.')
    else:
        print('   -> NOT consistent with zero. Do not trust the a_t channel below.')

    cm = {}
    raw = nfit.load_sweep('sweep_matched2d/log.mu*',
                          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    for key, rr in raw.items():
        cm[float(dict(key)['mu'])] = rr

    mus = sorted({r['mu_g'] for r in rows})
    pool = [cm[m] for m in mus if m in cm]
    T0, (lo, hi) = nfit.common_theta0(pool)
    print(f'\ncommon Theta_0 = {T0:.3e}\n')

    print('=' * 74)
    print('CHANNELS vs FRICTION   (2D, Pconf = 10)')
    print('=' * 74)
    print(f'{"mu_g":>7}{"C_c":>9}{"C_n":>9}{"C_t":>9}{"sum":>9}{"n fitted":>11}{"diff":>9}')
    out = []
    for mg in mus:
        rr = cm.get(mg)
        if not rr:
            continue
        Ts, _ = nfit.surviving_setpoints(rr)
        keep = {round(t, 12) for t in Ts}
        sel = [r for r in rows if r['mu_g'] == mg
               and round(r['Tgran'], 12) in keep and r['Theta'] > 0]
        if len(sel) < 8:
            continue
        x = np.log([r['Theta'] for r in sel]); x0 = np.log(T0)
        gm = local(x, [r['mu'] for r in sel], x0)
        if gm is None or gm[0] <= 0:
            continue
        C = {}
        ok = True
        for nm in ('a_c', 'a_n', 'a_t'):
            g = local(x, [r[nm] for r in sel], x0)
            if g is None:
                ok = False; break
            C[nm] = -g[1] / (2 * gm[0])
        if not ok:
            continue
        f = nfit.fit_local_sys(rr, T0)
        if not f:
            continue
        tot = sum(C.values())
        out.append((mg, C, tot, f))
        print(f'{mg:>7g}{C["a_c"]:>9.4f}{C["a_n"]:>9.4f}{C["a_t"]:>9.4f}'
              f'{tot:>9.4f}{f["n"]:>8.4f}+/-{f["tot"]:<4.3f}{tot-f["n"]:>9.4f}')

    if len(out) >= 3:
        d = np.array([o[2] - o[3]['n'] for o in out])
        e = np.array([o[3]['tot'] for o in out])
        print(f'\n   sum vs fitted n: mean {d.mean():+.4f}, RMS {np.sqrt((d**2).mean()):.4f}, '
              f'{np.sqrt(((d/e)**2).mean()):.1f} sigma')
        print('   (a_t is closure-derived here, so this is NOT an independent check')
        print('    of the identity -- it tests the DERIVATIVE decomposition only.)')

        print()
        print('=' * 74)
        print('WHERE DOES EACH CHANNEL PEAK?')
        print('=' * 74)
        mm = [o[0] for o in out]
        for k, nm in (('a_c', 'C_c'), ('a_n', 'C_n'), ('a_t', 'C_t')):
            v = [o[1][k] for o in out]
            print(f'   {nm}: peak at mu_g = {mm[int(np.argmax(v))]:g}   '
                  f'(range {min(v):+.4f} to {max(v):+.4f})')
        v = [o[2] for o in out]
        print(f'   n   : peak at mu_g = {mm[int(np.argmax(v))]:g}')


if __name__ == '__main__':
    main()
