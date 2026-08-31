"""
The Rothenburg-Bathurst channel decomposition of n.

In 2D the stress-fabric identity gives

    mu = (a_c + a_n + a_t) / 2

with a_c the fabric anisotropy, a_n the normal-force anisotropy and a_t the
tangential-force anisotropy. Checked directly against the contact dumps it
holds to 1.4% with 0.4% scatter over 10x in pressure, 10x in friction and 170x
in temperature -- far tighter than the ~0.025 precision floor on n itself.

Differentiating it turns the project's central quantity into a SUM:

    n = -dln(mu)/dln(Theta)
      = -[ d a_c/dlnTheta + d a_n/dlnTheta + d a_t/dlnTheta ] / (2 mu)
      =  C_c + C_n + C_t

So n is structurally a THREE-CHANNEL quantity. That is not a modelling choice;
it follows from an identity that the data satisfy to 1.4%. It also explains
the nine two-lever failures rather than merely restating them: no single
microstructural variable can be a state function for n unless the three
channels are slaved to one another.

WHY THE DERIVATIVE FORM, NOT THE WEIGHTED-LOG FORM. The equivalent expression
n = sum_i w_i n_i with w_i = a_i/(2 mu) and n_i = -dln(a_i)/dlnTheta is
algebraically the same, but a_t is ~4e-4 at low friction, so ln(a_t) and its
slope are numerically hopeless there while the product w_t n_t stays finite.
Working with d a_i/dlnTheta directly avoids taking the log of a vanishing
quantity.

TWO ESTIMATOR TRAPS, both hit and fixed here:
  * the density-weighted Fourier sum 2 sum_k E_k f_k exp(2i th_k)/f0 returns
    (a_c + a_n), NOT a_n. Subtracting a_c is essential; without it the identity
    over-predicts mu by a stable ~1.3x, which reads deceptively like a constant
    prefactor rather than a bug.
  * a_t rides the SIN channel -- f_t(th) = f0 a_t sin2(th - th_t) -- so it is
    the imaginary part after rotating by exp(-2i th_stress). Taking the real
    part returns a_t ~ 0, which is physically wrong at high friction.

THE CHECK THIS SCRIPT EXISTS TO MAKE. An identity accurate to 1.4% in mu does
not automatically decompose mu's LOGARITHMIC DERIVATIVE to the same accuracy.
So the decomposition is validated against the independently fitted n(Theta_0)
before any physics is read off it.

Usage:  python3 analyze_rb.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit

CACHE = 'rb_channels.csv'


# extraction

def frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                n = int(f.readline())
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                rows = []
                for _ in range(n):
                    t = f.readline().split()
                    if len(t) >= 11:
                        rows.append([float(x) for x in t[:11]])
                if rows:
                    yield np.array(rows)


def rb_of(path, K=36):
    """(mu, a_c, a_n, a_t) averaged over frames, all from the dump itself."""
    acc = []
    for a in frames(path):
        live = np.linalg.norm(a[:, 6:8], axis=1) > 0
        if live.sum() < 100:
            continue
        b = a[live]
        d = b[:, 4:6]
        dn = d / np.linalg.norm(d, axis=1)[:, None]
        # stress from the same contacts; 1/A cancels in mu = tau/P
        S = np.einsum('ci,cj->ij', b[:, 6:8] + b[:, 8:10], d)
        S = 0.5 * (S + S.T)
        P = (S[0, 0] + S[1, 1]) / 2.0
        if P <= 0:
            continue
        dev = complex((S[0, 0] - S[1, 1]) / 2.0, S[0, 1])
        ths = 0.5 * np.angle(dev)
        th = np.arctan2(dn[:, 1], dn[:, 0]) % np.pi
        that = np.stack([-dn[:, 1], dn[:, 0]], axis=1)
        fs = np.einsum('ij,ij->i', b[:, 6:8], dn)
        ts = np.einsum('ij,ij->i', b[:, 8:10], that)
        idx = np.floor(th / np.pi * K).astype(int) % K
        cnt = np.bincount(idx, minlength=K).astype(float)
        if (cnt == 0).any():
            continue
        E = cnt / cnt.sum()
        fnb = np.array([fs[idx == k].mean() for k in range(K)])
        ftb = np.array([ts[idx == k].mean() for k in range(K)])
        tc = (np.arange(K) + 0.5) * np.pi / K
        f0 = np.average(fnb, weights=cnt)
        ph = np.exp(-2j * ths)
        ac = (2 * np.sum(E * np.exp(2j * tc)) * ph).real
        acn = (2 * np.sum(E * fnb * np.exp(2j * tc)) / f0 * ph).real
        an = acn - ac                      # deconvolve the density weighting
        at = -(2 * np.sum(E * ftb * np.exp(2j * tc)) / f0 * ph).imag
        acc.append((abs(dev) / P, ac, an, at))
    if not acc:
        return None
    return np.mean(acc, axis=0)


def build_cache(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('P', 'mu_g', 'Tgran', 'mu', 'a_c', 'a_n', 'a_t', 'Theta'):
                r[k] = float(r[k])
        return rows
    rx = re.compile(r'dump\.contacts\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                    r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    paths = sorted(glob.glob('sweep_pichi/dump.contacts.P*'))
    rows = []
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        log = f'sweep_pichi/log.P{m.group("P")}_mu{m.group("mu")}_T{m.group("T")}_s{m.group("s")}'
        q = nfit.parse_log(log)
        if not q:
            continue
        got = rb_of(p)
        if got is None:
            continue
        P = float(m.group('P'))
        rows.append(dict(P=P, mu_g=float(m.group('mu')),
                         Tgran=float(m.group('T')) * P / 10.0, seed=m.group('s'),
                         mu=got[0], a_c=got[1], a_n=got[2], a_t=got[3],
                         Theta=q['Theta']))
        if (i + 1) % 50 == 0:
            print(f'   ... {i+1}/{len(paths)}', file=sys.stderr)
    if rows:
        with open(CACHE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


# local value and slope of a channel

def local(x, y, x0, deg=2):
    """Value and d/dx of y(x) at x0, from a local polynomial fit."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < deg + 2:
        return None
    A = np.vander(x - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = max(len(x) - (deg + 1), 1)
    cov = (float(r @ r) / dof) * np.linalg.inv(A.T @ A)
    return float(c[0]), float(c[1]), float(np.sqrt(cov[1, 1]))


def main():
    rows = build_cache('--recache' in sys.argv)
    if not rows:
        print('no dumps found')
        return
    print(f'{len(rows)} runs with RB channels\n')

    # identity check first: it underwrites everything below
    pred = np.array([(r['a_c'] + r['a_n'] + r['a_t']) / 2 for r in rows])
    meas = np.array([r['mu'] for r in rows])
    ratio = pred / meas
    print('=' * 78)
    print('IDENTITY CHECK   mu =? (a_c + a_n + a_t)/2')
    print('=' * 78)
    print(f'   ratio {ratio.mean():.4f} +/- {ratio.std():.4f} over {len(ratio)} runs'
          f'   (min {ratio.min():.3f}, max {ratio.max():.3f})')

    cells = {}
    for r in rows:
        cells.setdefault((r['P'], r['mu_g']), []).append(r)

    # gate, then decompose at a common reduced Theta_0
    runmap = nfit.load_sweep(
        'sweep_pichi/log.P*',
        r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
        tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    cm = {}
    for key, rr in runmap.items():
        d = dict(key)
        cm[(float(d['P']), float(d['mu']))] = rr

    los, his = [], []
    for (P, mg), rr in cm.items():
        Ts, byT = nfit.surviving_setpoints(rr)
        if len(Ts) < 4:
            continue
        th = [x['Theta'] / P for t in Ts for x in byT[t] if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    T0h = float(np.sqrt(max(los) * min(his)))

    print()
    print('=' * 78)
    print('CHANNEL DECOMPOSITION   n = C_c + C_n + C_t')
    print('=' * 78)
    print(f'   common reduced Theta_0 = {T0h:.3e}\n')
    print(f'{"P":>5}{"mu_g":>7}{"C_c":>9}{"C_n":>9}{"C_t":>9}{"sum":>9}'
          f'{"n (fitted)":>12}{"diff":>9}')
    out = []
    for (P, mg) in sorted(cells):
        rr = cm.get((P, mg))
        if not rr:
            continue
        Ts, _ = nfit.surviving_setpoints(rr)
        keep = {round(t, 12) for t in Ts}
        sel = [r for r in cells[(P, mg)] if round(r['Tgran'], 12) in keep
               and r['Theta'] > 0]
        if len(sel) < 8:
            continue
        T0 = T0h * P
        x = np.log([r['Theta'] for r in sel])
        x0 = np.log(T0)
        ch = {}
        for nm in ('a_c', 'a_n', 'a_t'):
            g = local(x, [r[nm] for r in sel], x0)
            if g is None:
                break
            ch[nm] = g
        if len(ch) < 3:
            continue
        gm = local(x, [r['mu'] for r in sel], x0)
        mu0 = gm[0]
        C = {nm: -ch[nm][1] / (2 * mu0) for nm in ch}
        Ce = {nm: ch[nm][2] / (2 * mu0) for nm in ch}
        tot = sum(C.values())
        f = nfit.fit_local_sys(rr, T0)
        if not f:
            continue
        out.append(((P, mg), C, Ce, tot, f))
        print(f'{P:>5g}{mg:>7g}{C["a_c"]:>9.4f}{C["a_n"]:>9.4f}{C["a_t"]:>9.4f}'
              f'{tot:>9.4f}{f["n"]:>9.4f}{"+/-"}{f["tot"]:<5.3f}'
              f'{tot - f["n"]:>9.4f}')

    if out:
        d = np.array([o[3] - o[4]['n'] for o in out])
        e = np.array([o[4]['tot'] for o in out])
        print(f'\n   decomposition vs fitted n: mean difference {d.mean():+.4f}, '
              f'RMS {np.sqrt((d**2).mean()):.4f}')
        print(f'   in units of the fitted error: RMS {np.sqrt(((d/e)**2).mean()):.1f} sigma')
        if np.sqrt(((d / e) ** 2).mean()) < 2:
            print('   -> the decomposition REPRODUCES n. Channel physics below is usable.')
        else:
            print('   -> the decomposition does NOT reproduce n. The identity holds for')
            print('      mu but not for its logarithmic derivative; do not read physics')
            print('      off the channels until this is understood.')

        print()
        print('=' * 78)
        print('WHICH CHANNEL CARRIES THE ERASURE?')
        print('=' * 78)
        print(f'{"P":>5}{"mu_g":>7}{"C_c/n":>9}{"C_n/n":>9}{"C_t/n":>9}   dominant')
        for (P, mg), C, Ce, tot, f in out:
            if abs(tot) < 1e-9:
                continue
            fr = {k: v / tot for k, v in C.items()}
            dom = max(fr, key=lambda k: abs(fr[k]))
            print(f'{P:>5g}{mg:>7g}{fr["a_c"]:>9.3f}{fr["a_n"]:>9.3f}'
                  f'{fr["a_t"]:>9.3f}   {dom}')


if __name__ == '__main__':
    main()
