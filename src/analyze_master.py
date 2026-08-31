"""
What functional form does n actually take?

Rather than guessing a formula and checking it, this fits a nested family of
models to every exponent measured at Pconf=10 and compares them by BIC. The
friction dependence is left NONPARAMETRIC -- each mu_g gets its own level --
so the comparison asks only how dimension and stiffness MODIFY the friction
curve, without committing to a shape for it. Committing to a shape first is
how you talk yourself into a formula the data does not support.

Models (f(mu_g) = one free level per friction value):
  A   n = f(mu_g)                                   friction alone
  B   n = f(mu_g) + a(D-2)                          + additive dimension
  C   n = f(mu_g) + b ln(kappa/kappa0)              + additive stiffness
  D   n = f(mu_g) + a(D-2) + b ln(kappa/kappa0)     both, additive
  E   n = f(mu_g) + a(D-2) + b(mu_g) ln(kappa/..)   stiffness coupling varies
                                                     with friction

Model E is the one that matters: the Emod sweep found dn/dln(E) = +0.002 at
mu_g=0.1 (flat) but +0.017 at mu_g=0.3 (9.8 sigma), so the stiffness coupling
is NOT a single constant. If E wins on BIC despite its extra parameters, the
exponent cannot be written as [friction term] + [stiffness term]; the two
interact, and any formula must carry a cross term.

Only Pconf=10 rows are used. Pressure has changed conclusions repeatedly in
this project, so the Pconf=2 k_t grid is excluded rather than pooled.

Usage:  python3 analyze_master.py [--csv master_n.csv]
"""
import argparse
import csv
import numpy as np

KAPPA0 = 1.0e4       # reference: Emod=1e5 at Pconf=10


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            row = dict(source=r['source'], D=float(r['D']),
                       mu_g=float(r['mu_g']), Pconf=float(r['Pconf']),
                       kappa=float(r['kappa']), kt_kn=float(r['kt_kn']),
                       n=float(r['n']), n_err=float(r['n_err']))
        except ValueError:
            continue
        if row['n_err'] > 0:
            out.append(row)
    return out


def wls(X, y, w):
    """Weighted least squares; returns coefficients and chi2."""
    Xw = X * w[:, None]
    yw = y * w
    c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    r = (X @ c - y) * w
    return c, float(r @ r)


def bic(chi2, k, N):
    return chi2 + k * np.log(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='master_n.csv')
    ap.add_argument('--default-kt-only', action='store_true',
                    help='restrict to the default k_t/k_n (the original fit)')
    args = ap.parse_args()

    rows = [r for r in load(args.csv) if r['Pconf'] == 10.0]
    ktd = max(set(r['kt_kn'] for r in rows), key=lambda v:
              sum(1 for r in rows if abs(r['kt_kn'] - v) < 1e-9))
    if args.default_kt_only:
        rows = [r for r in rows if abs(r['kt_kn'] - ktd) < 1e-9]

    mus = sorted({r['mu_g'] for r in rows})
    N = len(rows)
    print(f'{N} exponents at Pconf=10, k_t/k_n={ktd:.3f}, '
          f'{len(mus)} friction values, D in {sorted({int(r["D"]) for r in rows})}')
    print(f'kappa spans {min(r["kappa"] for r in rows):.0e} to '
          f'{max(r["kappa"] for r in rows):.0e}\n')

    y = np.array([r['n'] for r in rows])
    w = np.array([1.0 / r['n_err'] for r in rows])
    lk = np.array([np.log(r['kappa'] / KAPPA0) for r in rows])
    dD = np.array([r['D'] - 2.0 for r in rows])
    F = np.array([[1.0 if abs(r['mu_g'] - m) < 1e-12 else 0.0 for m in mus]
                  for r in rows])

    designs = {
        'A  f(mu_g)': F,
        'B  f + a(D-2)': np.hstack([F, dD[:, None]]),
        'C  f + b ln k': np.hstack([F, lk[:, None]]),
        'D  f + a(D-2) + b ln k': np.hstack([F, dD[:, None], lk[:, None]]),
    }
    # F: dimension enters MULTIPLICATIVELY on a friction ENHANCEMENT above a
    # shared baseline:  n = n0 + A_D * g(mu_g) + b ln kappa,  g(0)=0, A_2=1.
    # Motivated by the data rather than convenience: n(mu_g=0) is 0.0619 in 2D
    # and 0.0616 in 3D (indistinguishable), while the peaks differ by ~45%. A
    # constant additive dimension shift (model B) cannot express "identical at
    # zero friction, different at the peak", so B's failure does NOT establish
    # that dimension is irrelevant -- it only rejects a constant offset.
    # A_3 enters bilinearly, so it is profiled over a grid.
    mus_nz = [m for m in mus if m > 0]
    Gnz = np.array([[1.0 if abs(r['mu_g'] - m) < 1e-12 else 0.0 for m in mus_nz]
                    for r in rows])
    A_of_D = None
    bestF = None
    for A3 in np.linspace(0.5, 3.0, 251):
        scale = np.where(dD > 0, A3, 1.0)[:, None]
        XF = np.hstack([np.ones((N, 1)), Gnz * scale, lk[:, None]])
        cF, chi2F = wls(XF, y, w)
        if bestF is None or chi2F < bestF[1]:
            bestF, A_of_D = (cF, chi2F, XF), A3
    designs['F  n0 + A_D g(mu_g) + b ln k'] = bestF[2]

    varying = [m for m in mus
               if len({r['kappa'] for r in rows if r['mu_g'] == m}) > 1]
    if varying:
        Elk = np.array([[lk[i] if abs(rows[i]['mu_g'] - m) < 1e-12 else 0.0
                         for m in varying] for i in range(N)])
        designs['E  f + a(D-2) + b(mu_g) ln k'] = np.hstack([F, dD[:, None], Elk])

    # G: add the TANGENTIAL stiffness as its own variable. The matched-pressure
    # scans give dn/dln k_n > 0 but dn/dln k_t <= 0 -- opposite signs -- so a
    # single "stiffness" cannot describe n. If G beats F, that is the formal
    # statement of it.
    lkt = np.array([np.log(r['kt_kn'] / ktd) for r in rows])
    if np.ptp(lkt) > 0:
        bestG = None
        for A3 in np.linspace(0.5, 3.0, 251):
            XG = np.hstack([np.ones((N, 1)),
                            Gnz * np.where(dD > 0, A3, 1.0)[:, None],
                            lk[:, None], lkt[:, None]])
            cG, chi2G = wls(XG, y, w)
            if bestG is None or chi2G < bestG[1]:
                bestG, A3G = (cG, chi2G, XG), A3
        designs['G  F + c ln(k_t/k_n)'] = bestG[2]
        globals()['_A3G'] = A3G

    print(f'(model F profiled A_3 = {A_of_D:.3f})\n')
    print(f'{"model":<32}{"k":>4}{"chi2":>10}{"chi2/dof":>10}{"BIC":>10}')
    res = {}
    for name, X in designs.items():
        c, chi2 = wls(X, y, w)
        k = X.shape[1] + (1 if name[0] in 'FG' else 0)  # profiled A_3 costs a param
        res[name] = (bic(chi2, k, N), c, chi2, k)
        print(f'{name:<32}{k:>4}{chi2:>10.1f}{chi2/max(N-k,1):>10.2f}'
              f'{bic(chi2,k,N):>10.1f}')

    best = min(res, key=lambda kk: res[kk][0])
    b0 = res[best][0]
    print(f'\nbest by BIC: {best}')
    for name in designs:
        d = res[name][0] - b0
        tag = '   <-- best' if name == best else (
            '   decisively worse' if d > 10 else '   comparable' if d < 2 else '')
        print(f'   {name:<32} dBIC = {d:+7.1f}{tag}')

    c = res[best][1]
    print(f'\nfitted friction curve f(mu_g):')
    for m, v in zip(mus, c[:len(mus)]):
        print(f'   mu_g={m:<6g} f = {v:+.4f}')
    extra = c[len(mus):]
    if best[0] in 'BDE':
        print(f'   dimension term a = {extra[0]:+.4f} (2D -> 3D shift)')
    if best[0] == 'C':
        print(f'   stiffness term b = {extra[0]:+.4f} per e-fold in kappa')
    if best[0] == 'D':
        print(f'   stiffness term b = {extra[1]:+.4f} per e-fold in kappa')
    if best[0] == 'F':
        print(f'   baseline n0 = {c[0]:+.4f}   A_3/A_2 = {A_of_D:.3f}'
              f'   b = {c[-1]:+.4f} per e-fold in kappa')
        print('   friction enhancement g(mu_g), 2D units:')
        for m, v in zip([m for m in mus if m > 0], c[1:-1]):
            print(f'      mu_g={m:<6g} g = {v:+.4f}')
    if best[0] == 'G':
        print(f'   baseline n0 = {c[0]:+.4f}   A_3/A_2 = {globals().get("_A3G", float("nan")):.3f}')
        print(f'   b (normal/kappa)     = {c[-2]:+.5f} per e-fold')
        print(f'   c (tangential ratio) = {c[-1]:+.5f} per e-fold')
    if best[0] == 'E':
        print('   stiffness slope per friction:')
        for m, v in zip(varying, extra[1:]):
            print(f'      mu_g={m:<6g} b = {v:+.4f} per e-fold in kappa')


if __name__ == '__main__':
    main()
