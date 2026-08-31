"""The pressure lever at strain 1.0 -- constraints 9, 7a and 12, re-founded.

All three rested on strain-0.12 runs, and they are the results most exposed to
the transient: b = dn/dln(kappa) is a derivative with respect to stiffness, and
the crossings are friction-axis locations where a derivative changes sign, both
sitting on a friction-dependent bias.

Three things this campaign fixes beyond the strain:
  * chi is measured on the SAME runs that give n (in.granular_2d_piscan dumped
    nothing, which is why section 7a needed a separate 432-run campaign and had
    to match two pipelines).
  * Z is recorded, so the jamming gate applies.
  * four pressures, equal STRAIN in every cell (step counts computed per cell
    from gdot*dt), so the arms are not compared at different fractions of their
    own convergence.

Usage:  python3 analyze_pressure_steady.py
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit

D = 'sweep_lev_P'
RXL = r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
RXD = re.compile(r'dump\.contacts\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                 r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
CACHE = 'lev_P_chi.csv'
MOB = 0.99
ZMIN = 3.0                      # frictional isostatic floor in 2D


def load():
    raw = nfit.load_sweep(f'{D}/log.P*', RXL,
                          tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    out = {}
    for k, v in raw.items():
        d = dict(k)
        out[(float(d['P']), float(d['mu']))] = v
    return out


def jammed(runs):
    Ts, byT = nfit.surviving_setpoints(runs)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if not z or np.mean(z) >= ZMIN:
            keep.append(t)
    return keep


def chi_of(path, mu_g):
    from analyze_pichi import frames
    vals = []
    for a in frames(path):
        fn = np.linalg.norm(a[:, 6:8], axis=1)
        live = fn > 0
        if live.sum() < 100:
            continue
        vals.append(float((a[live, 10] / (mu_g * fn[live]) >= MOB).mean()))
    return float(np.mean(vals)) if vals else None


def build_chi(recache=False):
    if os.path.exists(CACHE) and not recache:
        rows = list(csv.DictReader(open(CACHE)))
        for r in rows:
            for k in ('P', 'mu_g', 'Tgran', 'chi'):
                r[k] = float(r[k])
        return rows
    rows = []
    paths = sorted(glob.glob(f'{D}/dump.contacts.P*'))
    for i, p in enumerate(paths):
        m = RXD.match(os.path.basename(p))
        if not m:
            continue
        P = float(m.group('P')); mg = float(m.group('mu'))
        c = chi_of(p, mg)
        if c is None:
            continue
        rows.append(dict(P=P, mu_g=mg, Tgran=float(m.group('T')) * P / 10.0,
                         seed=m.group('s'), chi=c))
        if (i + 1) % 60 == 0:
            print(f'   ... chi {i+1}/{len(paths)}', file=sys.stderr)
    with open(CACHE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def wline(x, y, e):
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(1 / np.asarray(e) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float((((y - A @ c) / e) ** 2).sum()) / max(len(x) - 2, 1)
    return c, cov, chi2


def main():
    cm = load()
    Ps = sorted({p for p, _ in cm})
    mus = sorted({m for _, m in cm})
    print(f'pressures {Ps}, frictions {mus}\n')

    # common REDUCED Theta_0 across all pressures, on jammed setpoints
    keep = {k: jammed(v) for k, v in cm.items()}
    los, his = [], []
    for (P, mg), ts in keep.items():
        if len(ts) < 4:
            continue
        _, byT = nfit.surviving_setpoints(cm[(P, mg)], restrict_to=ts)
        th = [x['Theta'] / P for t in ts for x in byT.get(t, []) if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    T0h = float(np.sqrt(max(los) * min(his)))
    print(f'common reduced Theta_0 = {T0h:.3e}\n')

    N = {}
    print(f'{"mu_g":>6}' + ''.join(f'{"P="+format(p,"g"):>20}' for p in Ps))
    for mg in mus:
        row = f'{mg:>6g}'
        for P in Ps:
            ts = keep.get((P, mg), [])
            r = nfit.fit_local_sys(cm[(P, mg)], T0h * P, restrict_to=ts) \
                if len(ts) >= 4 else None
            if r:
                N[(P, mg)] = r
                row += f'{r["n"]:>+13.4f}+/-{r["tot"]:<6.4f}'
            else:
                row += f'{"VOID":>20}'
        print(row)

    # constraint 9: b = dn/dln(kappa), kappa = E/P
    print()
    print('=' * 74)
    print('CONSTRAINT 9   b = dn/dln(kappa)   [kappa = E/P, so ln kappa = -ln P + c]')
    print('=' * 74)
    bs = []
    for mg in mus:
        pts = [(P, N[(P, mg)]) for P in Ps if (P, mg) in N]
        if len(pts) < 3:
            continue
        x = np.log(1.0e5 / np.array([p for p, _ in pts]))
        y = np.array([r['n'] for _, r in pts])
        e = np.array([r['tot'] for _, r in pts])
        c, cov, chi2 = wline(x, y, e)
        b, be = float(c[1]), float(np.sqrt(cov[1, 1]))
        bs.append((mg, b, be))
        print(f'   mu_g = {mg:<5g} b = {b:+.4f} +/- {be:.4f}  ({abs(b)/be:.1f} s)'
              f'   chi2/dof = {chi2:.2f}   [{len(pts)} pressures]')
    if len(bs) >= 3:
        x = np.log([m for m, _, _ in bs])
        y = np.array([b for _, b, _ in bs]); e = np.array([s for _, _, s in bs])
        c, cov, chi2 = wline(x, y, e)
        sl, sle = float(c[1]), float(np.sqrt(cov[1, 1]))
        print(f'\n   slope of b vs ln(mu_g): {sl:+.4f} +/- {sle:.4f} '
              f'({abs(sl)/sle:.1f} sigma), chi2/dof {chi2:.2f}')
        if sl != 0:
            xc = -c[0] / c[1]
            J = np.array([-1 / c[1], c[0] / c[1] ** 2])
            xce = float(np.sqrt(J @ cov @ J))
            print(f'   b crosses zero at mu_g = {np.exp(xc):.4f} '
                  f'(x/{np.exp(xce):.2f})     [strain-0.12 value: 0.1224 +/- 0.0254]')

    # constraint 7a: is chi a state function, with chi measured here
    print()
    print('=' * 74)
    print('SECTION 7a   n at matched chi, across a 10x pressure lever')
    print('=' * 74)
    rows = build_chi()
    print(f'   chi from {len(rows)} dumps of the same runs (no pipeline matching)')
    CH = {}
    for (P, mg), ts in keep.items():
        if len(ts) < 4:
            continue
        kp = {round(t, 12) for t in ts}
        v = [r['chi'] for r in rows if r['P'] == P and r['mu_g'] == mg
             and round(r['Tgran'], 12) in kp]
        if v:
            CH[(P, mg)] = float(np.mean(v))
    lo, hi = Ps[0], Ps[-1]
    A = sorted((CH[(lo, m)], N[(lo, m)]['n'], N[(lo, m)]['tot'], m)
               for m in mus if (lo, m) in CH and (lo, m) in N)
    B = [(CH[(hi, m)], N[(hi, m)]['n'], N[(hi, m)]['tot'], m)
         for m in mus if (hi, m) in CH and (hi, m) in N]
    fx = np.array([a[0] for a in A]); fy = np.array([a[1] for a in A])
    fe = np.array([a[2] for a in A])
    d, ee = [], []
    print(f'\n   reference P = {lo:g}; testing P = {hi:g}')
    for c_, n_, e_, mg in sorted(B):
        if c_ < fx.min() or c_ > fx.max():
            continue
        pred = float(np.interp(c_, fx, fy)); perr = float(np.interp(c_, fx, fe))
        d.append(n_ - pred); ee.append(float(np.hypot(e_, perr)))
        print(f'      mu_g={mg:<5g} chi={c_:.4f}  n={n_:+.4f}  predicted {pred:+.4f}'
              f'   {(n_-pred)/np.hypot(e_,perr):+.1f} sigma')
    if len(d) >= 3:
        d = np.array(d); ee = np.array(ee)
        mo = float(d.mean())
        se = float(np.sqrt((ee ** 2).mean() / len(d)))
        print(f'\n   coherent offset {mo:+.4f} +/- {se:.4f} -> {abs(mo)/se:.1f} sigma'
              f'   ({int((d<0).sum())}/{len(d)} negative)')
        print(f'   [strain-0.12 result: -0.0332 +/- 0.0085, 3.9 sigma]')


if __name__ == '__main__':
    main()
