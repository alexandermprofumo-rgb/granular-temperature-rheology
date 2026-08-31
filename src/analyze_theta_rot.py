"""ANALYSIS_PROTOCOL section 5.4 -- the LAST pre-registered test.

Theta_rot is a candidate state variable and gets exactly the same test as the
nine already falsified. Quoting the protocol verbatim:

    Statistic: the same two-lever/collapse machinery. Does n collapse against
    Theta_rot across levers that move Theta_rot by an orthogonal route?
    Pass: chi2/dof ~ 1 for a single n(Theta_rot) curve AND no residual trend
    with the lever variable at >= 2 sigma. Both conditions, as for chi.
    Prediction on record: fail.

WHICH LEVERS ARE AVAILABLE. `compute erotate/sphere` was added to the decks
part-way through, so only three sweeps carry v_Trot:

    sweep_size2d    N        = 1000 ... 32000      (199 runs)
    sweep_restit    e        = 0.1, 0.5, 0.9       (163 runs)
    sweep_iscan2    gdot     = 3.162e-4, 3.162e-3  (177 runs)

The friction lever lives inside each of them (mu_g = 0, 0.15, 1.0, and 0.3 in
the I-scan), so the pooled set is mu_g x {N, e, I} -- three orthogonal routes
to moving Theta_rot, which is what the protocol asks for. The friction,
stiffness, pressure and k_t sweeps predate the compute and cannot contribute;
that is a limitation of the test, not a choice made after seeing results.

A STRUCTURAL PROBLEM WITH THE CANDIDATE, worth stating before any number:
frictionless grains never spin up, so Theta_rot is identically ZERO at
mu_g = 0. ln(Theta_rot) is undefined there and every frictionless cell leaves
the fit. But n(mu_g = 0) ~ 0.035 is not zero -- so Theta_rot takes one value
where n takes a range, and it cannot organise the frictionless-to-frictional
transition that is the project's central structure (executive summary item 3).
Whatever the chi2 says, that alone caps what a pass could mean.

Both the raw Theta_rot and the dimensionless ratio Theta_rot/Theta are tested,
since the latter is the more natural state variable.

Usage:  python3 analyze_theta_rot.py
"""
import glob
import os
import re

import numpy as np

import nfit

ZMIN = 3.0

SWEEPS = [
    ('size', 'sweep_size2d',
     r'log\.N(?P<lev>[0-9]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     'N', np.log),
    ('restit', 'sweep_restit',
     r'log\.e(?P<lev>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     'e', lambda v: v),
    ('iscan', 'sweep_iscan2',
     r'log\.g(?P<lev>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     'gdot', np.log),
]


def parse_extra(path):
    """parse_log plus the measurement-window mean of v_Trot."""
    q = nfit.parse_log(path)
    if q is None:
        return None
    L = open(path, errors='ignore').readlines()
    i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT WINDOW' in l), None)
    if i is None:
        return q
    h, rows = None, []
    for l in L[i:]:
        t = l.split()
        if t and t[0] == 'Step':
            h = t
            continue
        if h is None:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            if rows:
                break
            continue
        if len(v) == len(h):
            rows.append(v)
    if rows and h and 'v_Trot' in h:
        a = np.array(rows)[len(rows) // 3:]
        q['Trot'] = float(a[:, h.index('v_Trot')].mean())
    return q


def load():
    out = {}
    for tag, sweep, rx_s, levname, levf in SWEEPS:
        rx = re.compile(rx_s)
        for p in glob.glob(f'{sweep}/log.*'):
            m = rx.match(os.path.basename(p))
            if not m:
                continue
            q = parse_extra(p)
            if not q or 'Trot' not in q:
                continue
            q['Tgran'] = float(m.group('T'))
            key = (tag, levf(float(m.group('lev'))), float(m.group('mu')))
            out.setdefault(key, []).append(q)
    return out


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= ZMIN]


def collapse(pts, xkey):
    """One curve y(ln x), quadratic; return chi2/dof and per-lever trends."""
    x = np.log([p[xkey] for p in pts])
    y = np.array([p['n'] for p in pts])
    e = np.array([p['e'] for p in pts])
    M = np.vstack([np.ones_like(x), x, x ** 2]).T
    W = np.diag(1 / e ** 2)
    c = np.linalg.inv(M.T @ W @ M) @ M.T @ W @ y
    res = y - M @ c
    chi2 = float(((res / e) ** 2).sum()) / max(len(x) - 3, 1)
    trends = {}
    for tag in sorted({p['tag'] for p in pts}):
        sel = np.array([p['tag'] == tag for p in pts])
        if sel.sum() < 3:
            continue
        lv = np.array([p['lev'] for p in pts])[sel]
        if np.std(lv) == 0:
            continue
        Ml = np.vstack([np.ones_like(lv), lv]).T
        Wl = np.diag(1 / e[sel] ** 2)
        cov = np.linalg.inv(Ml.T @ Wl @ Ml)
        cl = cov @ Ml.T @ Wl @ res[sel]
        trends[tag] = (abs(cl[1]) / np.sqrt(cov[1, 1]), int(sel.sum()))
    return chi2, trends


def main():
    cells = load()
    print('=' * 76)
    print('SECTION 5.4   Theta_rot as a state variable for n')
    print('           (pre-registered; prediction on record: FAIL)')
    print('=' * 76)
    nrun = sum(len(v) for v in cells.values())
    print(f'  {len(cells)} cells, {nrun} runs, from '
          + ', '.join(sorted({k[0] for k in cells})) + '\n')

    allth = [r['Theta'] for v in cells.values() for r in v if r['Theta'] > 0]
    T0 = float(np.exp(np.mean(np.log(allth))))
    print(f'  common Theta_0 = {T0:.3e}\n')

    pts, dropped, zero_rot = [], 0, 0
    print(f'  {"cell":<28}{"n":>18}{"Trot/Theta":>13}{"Trot":>12}')
    print('  ' + '-' * 71)
    for key in sorted(cells, key=str):
        tag, lev, mg = key
        runs = cells[key]
        ts = jammed(runs)
        if len(ts) < 4:
            dropped += 1
            continue
        _, byT = nfit.surviving_setpoints(runs, z_min=ZMIN, restrict_to=ts)
        th = [r['Theta'] for t in ts for r in byT.get(t, []) if r['Theta'] > 0]
        if not th or not (min(th) <= T0 <= max(th)):
            dropped += 1                     # Theta_0 not interior: no extrapolation
            continue
        f = nfit.fit_local_sys(runs, T0, restrict_to=ts, z_min=ZMIN)
        if f is None:
            dropped += 1
            continue
        tr = [r['Trot'] for t in ts for r in byT.get(t, []) if 'Trot' in r]
        rat = [r['Trot'] / r['Theta'] for t in ts for r in byT.get(t, [])
               if 'Trot' in r and r['Theta'] > 0]
        Trot = float(np.mean(tr)) if tr else 0.0
        ratio = float(np.mean(rat)) if rat else 0.0
        print(f'  {tag+" "+str(round(lev,4))+" mu"+str(mg):<28}'
              f'{f["n"]:>11.4f}+/-{f["tot"]:.4f}{ratio:>13.4f}{Trot:>12.3e}')
        if Trot <= 0 or ratio <= 0:
            zero_rot += 1
            continue
        pts.append(dict(tag=tag, lev=lev, mu_g=mg, n=f['n'], e=f['tot'],
                        Trot=Trot, ratio=ratio))

    print(f'\n  {len(pts)} cells enter the fit; {dropped} dropped (Theta_0 not '
          f'interior or too few\n  jammed setpoints); {zero_rot} dropped because '
          'Theta_rot = 0 exactly (mu_g = 0).')
    if len(pts) < 8:
        print('  too few cells to test.')
        return

    print()
    for xkey, lab in (('ratio', 'Theta_rot / Theta'), ('Trot', 'Theta_rot')):
        chi2, trends = collapse(pts, xkey)
        tmax = max((v[0] for v in trends.values()), default=0.0)
        ok = chi2 < 2.0 and tmax < 2.0
        print(f'  n against {lab}:')
        print(f'     single-curve chi2/dof = {chi2:.1f}'
              f'   ({"OK" if chi2 < 2 else "FAILS"} the <2 condition)')
        for tag, (tz, nc) in sorted(trends.items()):
            print(f'     residual trend vs the {tag:<7} lever: {tz:>5.1f} sigma'
                  f'   [{nc} cells]  {"ok" if tz < 2 else "FAILS"}')
        print(f'     -> {"** PASSES **" if ok else "FAILS"}\n')

    print('  ' + '-' * 71)
    print('  Multiplicity: Theta_rot is the ELEVENTH candidate state variable\n'
          '  tested for n. A pass would have to be read against that.')
    print('  Structural limit: Theta_rot = 0 identically at mu_g = 0, where\n'
          '  n ~ 0.035 -- so it cannot organise the frictionless-to-frictional\n'
          '  transition regardless of how the surviving cells behave.')


if __name__ == '__main__':
    main()
