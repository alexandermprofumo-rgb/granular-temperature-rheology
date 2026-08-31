"""
Two revalidations under the corrected error budget.

PART 1 -- the nine falsified organizing variables, re-tested with sigma_sys.
Every two-lever discrepancy was quoted against the STATISTICAL error alone.
sigma_sys (the Theta-window systematic) is typically ~2x sigma_stat, so a
3 sigma failure may be a 1.5 sigma failure. Falsifications that drop below
~2 sigma can no longer be claimed. The prediction interpolated from the
friction-lever curve also carries an error, included here via the two
bracketing points -- omitting it was a second way the old numbers were
optimistic.

PART 2 -- is the mu_g ~ 0.3 anomaly anything more than curvature?
The curvature (n rising with Theta) is largest at exactly mu_g = 0.2-0.5,
where the anomaly lives. If b(mu_g) = dn/dln(kappa) is a real feature it must
survive being computed from a FIXED part of the Theta window; if it is a
curvature artifact its shape will change when the window moves. So b(mu_g) is
recomputed three ways -- full window, low half only, high half only -- and
compared. A feature that inverts or vanishes between them is not physics.

Usage:  python3 revalidate.py
"""
import glob
import os
import re

import numpy as np

import nfit


def sysc(runs, res):
    """Window systematic, deconvolved (see recompute_constraints.sys_of)."""
    if res is None or not np.isfinite(res['sys']):
        return np.nan
    Ts = res['setpoints']
    if len(Ts) >= 6:
        h = len(Ts) // 2
        lo = nfit.fit_n(runs, restrict_to=Ts[:h + 1])
        hi = nfit.fit_n(runs, restrict_to=Ts[h:])
        if lo and hi:
            return abs(hi['n'] - lo['n']) / 2.0
    return float(np.sqrt(max(res['sys'] ** 2 - res['stat'] ** 2, 0.0)))


def full(runs, window=None):
    r = nfit.fit_n(runs, restrict_to=window)
    if r is None:
        return None
    s = sysc(runs, r)
    r['tot2'] = float(np.hypot(r['stat'], s)) if np.isfinite(s) else r['stat']
    return r


def load(pattern, rx, tg=None):
    return nfit.load_sweep(pattern, rx, tg)


def csv(path):
    import csv as _c
    rs = list(_c.DictReader(open(path)))
    for r in rs:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rs


W4 = [0.001, 0.003, 0.008, 0.02]


def part1():
    print('=' * 78)
    print('PART 1 -- two-lever falsifications with sigma_stat (+) sigma_sys')
    print('=' * 78)

    # friction lever (Emod = 1e5, Pconf = 10)
    fr = {}
    for key, runs in load('sweep_matched2d/log.mu*',
                          r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
        fr.setdefault(float(dict(key)['mu']), []).extend(runs)
    # stiffness lever
    st = {}
    for key, runs in load('sweep_stiffness/log.E*',
                          r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
        d = dict(key)
        st[(float(d['E']), float(d['mu']))] = runs
    # k_t lever
    kt = {}
    for key, runs in load('sweep_kt_P10/log.k*',
                          r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$').items():
        d = dict(key)
        kt[(float(d['kt']), float(d['mu']))] = runs

    micro_f = csv('tier1_matched2d_results.csv')
    micro_s = csv('tier1_stiff_results.csv')
    for r in micro_s:
        r['E'] = float(re.match(r'E([0-9.eE+-]+)_mu', r['label']).group(1))
    dyn_f = csv('dyn_fric.csv') if os.path.exists('dyn_fric.csv') else []
    dyn_k = csv('dyn_kt.csv') if os.path.exists('dyn_kt.csv') else []

    STATIC = ('chi', 'Z', 'dZ_eff', 'cv_f', 'pr_f', 'f_mean', 'a_c_w')
    print(f'\n{"variable":<10}{"lever":<12}{"old (stat only)":>18}{"new (stat+sys)":>18}   verdict')
    print('-' * 78)

    for q in STATIC:
        # curve from the friction lever
        pts = []
        for mg in sorted(fr):
            if mg == 0:
                continue
            r = full(fr[mg], W4)
            s = [x for x in micro_f if x['mu_g'] == mg]
            if r and s:
                pts.append((np.mean([x[q] for x in s]), r['n'], r['stat'], r['tot2']))
        if len(pts) < 4:
            continue
        pts.sort()
        fx = np.array([p[0] for p in pts]); fy = np.array([p[1] for p in pts])
        fe = np.array([p[3] for p in pts])
        old, new = [], []
        for (E, mg), runs in sorted(st.items()):
            if abs(E - 1e5) < 1:
                continue
            r = full(runs, W4)
            s = [x for x in micro_s if x['E'] == E and x['mu_g'] == mg]
            if not (r and s):
                continue
            x = np.mean([x[q] for x in s])
            if x < fx.min() or x > fx.max():
                continue
            pred = float(np.interp(x, fx, fy))
            perr = float(np.interp(x, fx, fe))          # curve uncertainty
            old.append((r['n'] - pred) / r['stat'])
            new.append((r['n'] - pred) / np.hypot(r['tot2'], perr))
        if len(new) < 3:
            continue
        o = np.sqrt(np.mean(np.array(old) ** 2))
        nn = np.sqrt(np.mean(np.array(new) ** 2))
        v = 'still FAILS' if nn >= 2 else 'NO LONGER significant'
        print(f'{q:<10}{"stiffness":<12}{o:>15.1f} s{nn:>15.1f} s   {v}')

    # R_E under the k_t lever (the strongest lever available)
    if dyn_f and dyn_k:
        pts = []
        for mg in sorted(fr):
            if mg == 0:
                continue
            r = full(fr[mg], W4)
            s = [x for x in dyn_f if x['mu_g'] == mg]
            if r and s:
                pts.append((np.mean([x['R_E'] for x in s]), r['n'], r['stat'], r['tot2']))
        pts.sort()
        fx = np.array([p[0] for p in pts]); fy = np.array([p[1] for p in pts])
        fe = np.array([p[3] for p in pts])
        old, new = [], []
        for (ktv, mg), runs in sorted(kt.items()):
            r = full(runs, W4)
            ktkn = ktv / 7.326e4
            s = [x for x in dyn_k if abs(x['kt_kn'] - ktkn) < 1e-6 and x['mu_g'] == mg]
            if not (r and s):
                continue
            x = np.mean([x['R_E'] for x in s])
            if x < fx.min() or x > fx.max():
                continue
            pred = float(np.interp(x, fx, fy))
            perr = float(np.interp(x, fx, fe))
            old.append((r['n'] - pred) / r['stat'])
            new.append((r['n'] - pred) / np.hypot(r['tot2'], perr))
        if len(new) >= 3:
            o = np.sqrt(np.mean(np.array(old) ** 2))
            nn = np.sqrt(np.mean(np.array(new) ** 2))
            v = 'still FAILS' if nn >= 2 else 'NO LONGER significant'
            print(f'{"R_E":<10}{"k_t":<12}{o:>15.1f} s{nn:>15.1f} s   {v}')


def part2():
    print()
    print('=' * 78)
    print('PART 2 -- does the mu_g~0.3 anomaly survive a window change?')
    print('=' * 78)
    cells = {}
    for key, runs in load(
            'sweep_piscan/log.P*',
            r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
            tg=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0).items():
        d = dict(key)
        cells[(float(d['P']), float(d['mu']))] = runs
    Ps = [5.0, 10.0, 25.0, 50.0]
    mus = sorted({k[1] for k in cells})

    def bof(mode):
        out = {}
        for mg in mus:
            xs, ns, es = [], [], []
            for P in Ps:
                if (P, mg) not in cells:
                    continue
                r = nfit.fit_n(cells[(P, mg)])
                if r is None:
                    continue
                Ts = r['setpoints']
                if mode == 'low':
                    Ts = Ts[:max(3, len(Ts) // 2 + 1)]
                elif mode == 'high':
                    Ts = Ts[-max(3, len(Ts) // 2 + 1):]
                rr = nfit.fit_n(cells[(P, mg)], restrict_to=Ts)
                if rr is None:
                    continue
                xs.append(np.log(1e5 / P / 1e4)); ns.append(rr['n']); es.append(rr['stat'])
            if len(xs) < 3:
                continue
            w = 1 / np.array(es)
            A = np.vstack([xs, np.ones(len(xs))]).T
            c, *_ = np.linalg.lstsq(A * w[:, None], np.array(ns) * w, rcond=None)
            out[mg] = float(c[0])
        return out

    fullb, lowb, highb = bof('full'), bof('low'), bof('high')
    print(f'\n{"mu_g":>7}{"b(full window)":>17}{"b(low Theta)":>15}{"b(high Theta)":>16}')
    common = [m for m in mus if m in fullb and m in lowb and m in highb]
    for mg in common:
        print(f'{mg:>7g}{fullb[mg]:>17.4f}{lowb[mg]:>15.4f}{highb[mg]:>16.4f}')
    if common:
        pk = lambda d: max(common, key=lambda m: d[m])
        print(f'\n   b peaks at mu_g = {pk(fullb):g} (full), {pk(lowb):g} (low Theta), '
              f'{pk(highb):g} (high Theta)')
        a = np.array([fullb[m] for m in common])
        b_ = np.array([lowb[m] for m in common])
        c_ = np.array([highb[m] for m in common])
        print(f'   corr(full, low)  = {np.corrcoef(a, b_)[0,1]:+.2f}')
        print(f'   corr(full, high) = {np.corrcoef(a, c_)[0,1]:+.2f}')
        print(f'   corr(low, high)  = {np.corrcoef(b_, c_)[0,1]:+.2f}')
        print('\n   A feature that keeps its position and shape across all three')
        print('   windows is physics. One that moves or inverts is curvature.')


if __name__ == '__main__':
    part1()
    part2()
