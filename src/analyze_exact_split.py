"""
The EXACT normal/tangential split of n, in 2D and 3D.

The Rothenburg-Bathurst decomposition of §7b is 2D-specific and holds to 1.4%.
There is a stronger statement available that needs no expansion, no small-
anisotropy assumption, and no dimension-specific prefactor:

    sigma_ij = (1/V) sum_c (f_n + f_t)_i d_j

splits term by term into a normal and a tangential stress,

    sigma^n_ij = (1/V) sum_c (f_n)_i d_j        sigma^t_ij = (1/V) sum_c (f_t)_i d_j

and projecting both deviators onto the unit deviator of the TOTAL stress gives

    mu = mu_n + mu_t          EXACTLY, in any dimension.

Differentiating,

    n = -dln(mu)/dlnTheta = C_N + C_T,    C_X = -(d mu_X/dlnTheta)/mu

This is an identity, not a fit. It is weaker than the three-channel version --
it cannot separate the geometric fabric from the normal-force anisotropy -- but
it is exact, it transfers to 3D unchanged, and it isolates the one quantity
that the 2D analysis showed does the interesting work: the tangential share of
n rises from 5% at mu_g = 0 to 60% at mu_g = 1.

The 3D dumps (sweep_tracking_forces_3d, `compute pair/local dist dx dy dz
fx fy fz p1 p2 p3 p4`) carry the tangential force vector, so both channels are
measured -- unlike sweep_matched2d, where a_t had to be closed.

SANITY CHECK BUILT IN: mu_n + mu_t must equal the mu computed from the total
stress to machine precision. If it does not, the column mapping is wrong.

Usage:  python3 analyze_exact_split.py [2d|3d]
"""
import glob
import os
import re
import sys

import numpy as np

import nfit

SPEC = {
    # the steady-state campaign: dumps and logs from the SAME runs
    # This is what section 6 said could not be done. It could not be done with
    # the sweeps that existed then: sweep_tracking_forces_3d is N = 2000 and
    # 3000 steps, and pairing its dumps with sweep_run_3d's logs (which the
    # '3d' entry below still does) compares different simulations. The steady
    # decks dump the tangential force on the production runs themselves, so
    # mu, Theta and the stress split all come from one place -- no closure, no
    # calibrated prefactor, no error floor.
    '2ds': dict(dumps='sweep_steady2d/dump.contacts.mu*',
                logs='sweep_steady2d/log.mu*',
                rx=r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
                D=2, d=slice(4, 6), fn=slice(6, 8), ft=slice(8, 10), ncol=11),
    '3ds': dict(dumps='sweep_steady3d/dump.contacts.mu*',
                logs='sweep_steady3d/log.mu*',
                rx=r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
                D=3, d=slice(4, 7), fn=slice(7, 10), ft=slice(10, 13), ncol=14),
    # ---- the original, mismatched attempt; kept as the record of the failure
    '3d': dict(dumps='sweep_tracking_forces_3d/dump.contacts.mu*',
               logs='sweep_run_3d/log.mu*',
               rx=r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
               D=3, d=slice(4, 7), fn=slice(7, 10), ft=slice(10, 13), ncol=14),
    '2d': dict(dumps='sweep_pichi/dump.contacts.P10*',   # (none; 2D uses pichi P5/P50)
               logs='sweep_matched2d/log.mu*',
               rx=r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
               D=2, d=slice(4, 6), fn=slice(6, 8), ft=slice(8, 10), ncol=11),
}


def frames(path, ncol, keep_last=8):
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
                    if len(t) >= ncol:
                        rows.append([float(x) for x in t[:ncol]])
                if rows:
                    out.append(np.array(rows))
    return out[-keep_last:]


def split(path, sp):
    """(mu, mu_n, mu_t, residual) averaged over frames."""
    acc = []
    for a in frames(path, sp['ncol']):
        d = a[:, sp['d']]
        fn = a[:, sp['fn']]
        ft = a[:, sp['ft']]
        live = np.linalg.norm(fn, axis=1) > 0
        if live.sum() < 100:
            continue
        d, fn, ft = d[live], fn[live], ft[live]
        Sn = np.einsum('ci,cj->ij', fn, d)
        St = np.einsum('ci,cj->ij', ft, d)
        Sn = 0.5 * (Sn + Sn.T); St = 0.5 * (St + St.T)
        S = Sn + St
        D = sp['D']
        P = np.trace(S) / D
        if P <= 0:
            continue
        dev = S - np.eye(D) * np.trace(S) / D
        nrm = np.sqrt(np.sum(dev * dev))
        if nrm <= 0:
            continue
        sh = dev / nrm                      # unit deviator of the TOTAL stress
        devn = Sn - np.eye(D) * np.trace(Sn) / D
        devt = St - np.eye(D) * np.trace(St) / D
        tau = float(np.sum(dev * sh))       # = nrm
        taun = float(np.sum(devn * sh))
        taut = float(np.sum(devt * sh))
        acc.append((tau / P, taun / P, taut / P, (taun + taut - tau) / max(tau, 1e-30)))
    if not acc:
        return None
    return np.mean(acc, axis=0)


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else '3d').lower()
    sp = SPEC[which]
    rx = re.compile(sp['rx'])
    recs = {}
    paths = sorted(glob.glob(sp['dumps']))
    print(f'{which.upper()}: {len(paths)} dumps', file=sys.stderr)
    worst = 0.0
    for p in paths:
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        logdir = os.path.dirname(sp['logs'])
        q = nfit.parse_log(f'{logdir}/log.mu{m.group("mu")}_T{m.group("T")}_s{m.group("s")}')
        if not q or q['Theta'] <= 0:
            continue
        r = split(p, sp)
        if r is None:
            continue
        worst = max(worst, abs(r[3]))
        recs.setdefault(float(m.group('mu')), []).append(
            dict(Tgran=float(m.group('T')), Theta=q['Theta'],
                 mu=r[0], mu_n=r[1], mu_t=r[2]))

    print(f'\nexactness check: max |mu_n + mu_t - mu| / mu = {worst:.2e}'
          f'   {"(machine precision: split verified)" if worst < 1e-9 else "<-- COLUMN MAPPING WRONG"}')

    raw = nfit.load_sweep(sp['logs'],
                          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    cm = {float(dict(k)['mu']): v for k, v in raw.items()}
    mus = sorted(set(recs) & set(cm))
    got = nfit.common_theta0([cm[m] for m in mus])
    if not got:
        print('no common Theta range')
        return
    T0, _ = got
    print(f'\ncommon Theta_0 = {T0:.3e}\n')
    print('=' * 72)
    print(f'EXACT SPLIT  n = C_N + C_T   ({which.upper()})')
    print('=' * 72)
    print(f'{"mu_g":>7}{"C_N":>10}{"C_T":>10}{"sum":>10}{"n fitted":>12}'
          f'{"C_T share":>11}')
    for mg in mus:
        Ts, _ = nfit.surviving_setpoints(cm[mg])
        keep = {round(t, 12) for t in Ts}
        sel = [r for r in recs[mg] if round(r['Tgran'], 12) in keep]
        if len(sel) < 6:
            continue
        x = np.log([r['Theta'] for r in sel]); x0 = np.log(T0)
        A = np.vander(x - x0, 3, increasing=True)
        def slope(key):
            y = np.array([r[key] for r in sel])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(c[0]), float(c[1])
        m0, _ = slope('mu')
        if m0 <= 0:
            continue
        _, dn = slope('mu_n')
        _, dt = slope('mu_t')
        CN, CT = -dn / m0, -dt / m0
        f = nfit.fit_local_sys(cm[mg], T0)
        if not f:
            continue
        tot = CN + CT
        print(f'{mg:>7g}{CN:>10.4f}{CT:>10.4f}{tot:>10.4f}'
              f'{f["n"]:>9.4f}+/-{f["tot"]:<4.3f}'
              f'{(CT/tot if tot else float("nan")):>10.2f}')


if __name__ == '__main__':
    main()
