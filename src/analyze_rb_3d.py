"""
The channel decomposition in 3D.

The production 3D sweep already writes per-contact dumps (in.granular_3d,
`compute pl all pair/local dx dy dz fx fy fz`), on the same N = 4000 runs whose
logs carry mu, Theta and the measurement-window marker. So the 3D decomposition
needs no new simulations -- the earlier conclusion that it did came from looking
at sweep_tracking_forces_3d, which is a separate N = 2000 tracking set.

DIMENSION-GENERAL FORMULATION. The 2D analysis used Fourier coefficients in the
contact angle, which does not transfer. The tensor form does, and avoids having
to import a dimension-specific prefactor:

    S_ij   = sum_c f_n^c d_j^c n_i^c        (normal contact stress, from dumps)
    s_hat  = unit deviator of S
    F_ij   = <n_i n_j>                      (fabric tensor)
    G_ij   = <f_n n_i n_j> / <f_n>          (force-weighted fabric)

    A_c = F' : s_hat                        geometric anisotropy
    A_n = (G' - F') : s_hat                 force anisotropy

G' contains BOTH the geometric and the force anisotropy -- that is the same
structure as the 2D estimator trap, where the density-weighted Fourier sum
returned (a_c + a_n) rather than a_n. Subtracting F' is the tensor version of
that correction.

THE PREFACTOR IS MEASURED, NOT ASSUMED. Rothenburg-Bathurst gives mu =
(2/5)(a_c + a_n + a_t) in 3D against (1/2)(...) in 2D, with normalisation
conventions that are easy to get wrong. Instead:

    at mu_g = 0 there is NO tangential force, so A_t = 0 exactly

and k is fixed by regressing mu (from the log) on (A_c + A_n) over the
frictionless cells alone. That is a measurement on data where the answer is
known a priori, and it doubles as a test: if mu is not proportional to
(A_c + A_n) at zero friction, the formulation is wrong and nothing else should
be believed.

A_t is then closed as mu/k - A_c - A_n, exactly as in the 2D fine-resolution
analysis, with the same caveat: the sum reproduces n by construction, so it
tests fitting consistency, not the identity.

Usage:  python3 analyze_rb_3d.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit
from analyze_rb import local

CACHE = 'rb_3d.csv'
NCOL = 7          # index, dx, dy, dz, fx, fy, fz
NFRAMES = 8


def frames(path, keep_last=NFRAMES):
    out = []
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
                    if len(t) >= NCOL:
                        rows.append([float(x) for x in t[:NCOL]])
                if rows:
                    out.append(np.array(rows))
    return out[-keep_last:]


def anis(path):
    """(A_c, A_n) projected on the stress axis, averaged over frames."""
    acc = []
    for a in frames(path):
        d = a[:, 1:4]
        f = a[:, 4:7]
        nrm = np.linalg.norm(d, axis=1)
        fm = np.linalg.norm(f, axis=1)
        live = (fm > 0) & (nrm > 0)
        if live.sum() < 200:
            continue
        d, f, nrm = d[live], f[live], nrm[live]
        n = d / nrm[:, None]
        fn = np.einsum('ij,ij->i', f, n)          # signed normal magnitude
        S = np.einsum('ci,cj->ij', f, d)
        S = 0.5 * (S + S.T)
        tr = np.trace(S)
        if tr <= 0:
            continue
        dev = S - np.eye(3) * tr / 3.0
        nd = np.sqrt(np.sum(dev * dev))
        if nd <= 0:
            continue
        sh = dev / nd
        F = np.einsum('ci,cj->ij', n, n) / len(n)
        w = fn / fn.sum()
        G = np.einsum('c,ci,cj->ij', w, n, n)
        Fd = F - np.eye(3) * np.trace(F) / 3.0
        Gd = G - np.eye(3) * np.trace(G) / 3.0
        acc.append((float(np.sum(Fd * sh)), float(np.sum((Gd - Fd) * sh))))
    if not acc:
        return None
    return np.mean(acc, axis=0)


def build(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('mu_g', 'Tgran', 'mu', 'A_c', 'A_n', 'Theta'):
                r[k] = float(r[k])
        return rows
    rx = re.compile(r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    paths = sorted(glob.glob('sweep_run_3d/dump.contacts.mu*'))
    rows = []
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(f'sweep_run_3d/log.mu{m.group("mu")}'
                           f'_T{m.group("T")}_s{m.group("s")}')
        if not q or q['mu'] <= 0 or q['Theta'] <= 0:
            continue
        got = anis(p)
        if got is None:
            continue
        rows.append(dict(mu_g=float(m.group('mu')), Tgran=float(m.group('T')),
                         seed=m.group('s'), mu=q['mu'], Theta=q['Theta'],
                         A_c=got[0], A_n=got[1]))
        if (i + 1) % 40 == 0:
            print(f'   ... {i+1}/{len(paths)}', file=sys.stderr)
    if rows:
        with open(CACHE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows


def main():
    rows = build('--recache' in sys.argv)
    if not rows:
        print('no 3D dumps parsed')
        return
    print(f'{len(rows)} 3D runs\n')

    print('=' * 74)
    print('CALIBRATION -- k from the frictionless cells, where A_t = 0 exactly')
    print('=' * 74)
    z = [r for r in rows if r['mu_g'] == 0]
    if len(z) < 5:
        print('   too few frictionless runs')
        return
    x = np.array([r['A_c'] + r['A_n'] for r in z])
    y = np.array([r['mu'] for r in z])
    k = float(np.sum(x * y) / np.sum(x * x))            # through the origin
    pred = k * x
    rel = pred / y
    print(f'   k = {k:.4f}   ({len(z)} frictionless runs)')
    print(f'   mu_pred/mu_meas = {rel.mean():.4f} +/- {rel.std():.4f}'
          f'   (min {rel.min():.3f}, max {rel.max():.3f})')
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f'   R^2 = {r2:.4f}')
    print(f'   RB would give 2/5 = 0.4000 under its own normalisation; the value')
    print(f'   here carries this code\'s conventions, so only its CONSTANCY matters.')
    if rel.std() > 0.05 or r2 < 0.8:
        print('   -> mu is NOT proportional to (A_c + A_n) at zero friction.')
        print('      The formulation is wrong; stop here.')
        return
    print('   -> proportional at zero friction; formulation validated.')

    for r in rows:
        r['A_t'] = r['mu'] / k - r['A_c'] - r['A_n']

    nz = [r['A_t'] for r in rows if r['mu_g'] >= 0.5]
    print(f'\n   A_t(mu_g=0)   = {np.mean([r["A_t"] for r in z]):+.4f} '
          f'+/- {np.std([r["A_t"] for r in z]):.4f}   (zero by construction of k)')
    if nz:
        print(f'   A_t(mu_g>=0.5) = {np.mean(nz):+.4f} +/- {np.std(nz):.4f}'
              f'   (must be clearly positive if friction adds tangential stress)')

    raw = nfit.load_sweep('sweep_run_3d/log.mu*',
                          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    cm = {float(dict(kk)['mu']): v for kk, v in raw.items()}
    mus = sorted(set(r['mu_g'] for r in rows) & set(cm))
    got = nfit.common_theta0([cm[m] for m in mus])
    if not got:
        print('no common Theta range')
        return
    T0, _ = got
    print(f'\ncommon Theta_0 = {T0:.3e}\n')
    print('=' * 74)
    print('3D CHANNELS   n = C_c + C_n + C_t')
    print('=' * 74)
    print(f'{"mu_g":>7}{"C_c":>9}{"C_n":>9}{"C_t":>9}{"sum":>9}{"n fitted":>12}')
    out = []
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
        C, ok = {}, True
        for nm in ('A_c', 'A_n', 'A_t'):
            g = local(x, [r[nm] for r in sel], x0)
            if g is None:
                ok = False; break
            C[nm] = -g[1] * k / gm[0]
        if not ok:
            continue
        f = nfit.fit_local_sys(cm[mg], T0)
        if not f:
            continue
        tot = sum(C.values())
        out.append((mg, C, tot, f))
        print(f'{mg:>7g}{C["A_c"]:>9.4f}{C["A_n"]:>9.4f}{C["A_t"]:>9.4f}'
              f'{tot:>9.4f}{f["n"]:>9.4f}+/-{f["tot"]:<5.3f}')

    if len(out) >= 3:
        # THE GATE. analyze_rb.py and analyze_rb_fine.py both compute this and
        # both refuse to read physics off the channels if it exceeds 2 sigma.
        # This script did not, which is how a decomposition that misses n by a
        # factor of 2.1 at mu_g = 0 -- the one cell where the answer is known a
        # priori, since A_t must vanish there -- came to be quoted.
        d = np.array([tot - f['n'] for _, _, tot, f in out])
        e = np.array([f['tot'] for _, _, _, f in out])
        rms = float(np.sqrt(((d / e) ** 2).mean()))
        print()
        print('=' * 74)
        print('CONSISTENCY GATE   sum C_i =? independently fitted n')
        print('=' * 74)
        print(f'   mean difference {d.mean():+.4f}, RMS {np.sqrt((d**2).mean()):.4f}')
        print(f'   in units of the fitted error: RMS {rms:.1f} sigma')
        worst = sorted(zip([o[0] for o in out], d / e), key=lambda t: -abs(t[1]))[:3]
        print('   worst cells: ' + ', '.join(f'mu_g={m:g} at {s:+.1f} sigma'
                                             for m, s in worst))
        if rms >= 2:
            print('   -> the decomposition does NOT reproduce n in 3D. The shares')
            print('      below are NOT usable as physics.')
        else:
            print('   -> marginal. Note the mismatch is SYSTEMATIC (positive at')
            print('      mu_g <= 0.2, ~zero above), not scatter, so the low-friction')
            print('      shares carry an unquantified bias even though the RMS passes.')

        print()
        print('=' * 74)
        print('CHANNEL SHARES -- does the 2D inversion repeat in 3D?')
        print('=' * 74)
        print(f'{"mu_g":>7}{"C_c/n":>9}{"C_n/n":>9}{"C_t/n":>9}')
        for mg, C, tot, f in out:
            if abs(tot) < 1e-9:
                continue
            print(f'{mg:>7g}{C["A_c"]/tot:>9.2f}{C["A_n"]/tot:>9.2f}{C["A_t"]/tot:>9.2f}')


if __name__ == '__main__':
    main()
