"""Does mu * Theta^n = F(I) hold AT ALL, as a functional form?

The relation has always been tested by measuring n and comparing it to 1/(2D).
That tests the VALUE. It never tests the FORM, and the form is the stronger
claim: writing

    mu(Theta, I) * Theta^n = F(I)     <=>     ln mu = ln F(I) - n ln Theta

asserts that Theta and I enter SEPARABLY -- that changing I slides the
ln mu vs ln Theta curve vertically without changing its shape. If the two arms
are not parallel, no value of n can rescue the relation, and the exponent
becomes a two-variable function n(Theta, I) that no longer means anything.

WHAT IS TESTED, on two arms 3.162x apart in I:

   arm A   gdot = 1.0e-3    I = 3.162e-4    sweep_steady2d   (11 frictions)
   arm B   gdot = 3.162e-3  I = 1.0e-3      sweep_iscan2     (4 frictions)

both at strain 1.0, same deck, jamming-gated. Pooled design, with
u = ln(Theta/Theta_0) and J = 1 on arm B:

   ln mu = a + s*J + b*u + beta*(J*u) + c*u^2 + gamma*(J*u^2)

   s      = ln F(I_B) - ln F(I_A)      the shape of F, if F exists
   beta   = -Delta n                   SEPARABILITY REQUIRES beta = 0
   gamma  = -Delta curvature / 2       SEPARABILITY REQUIRES gamma = 0

Errors are setpoint-jackknifed and calibrated against the analytic null in the
same way as every other number in the project (nfit._jk_null_factor_A), so
these are comparable to the constraint-list errors.

SECOND TEST, re-derived on steady data. If Theta and I combine into a single
variable at all, the natural group is R = v_T/(gdot d) -- thermal velocity over
shear velocity -- with v_T ~ sqrt(Theta P/rho), so R ~ sqrt(Theta)/I. A law
mu = G(R) with the observed local slope -n in Theta then forces

   d ln mu / d ln I  =  +2n .

An earlier measured ratio of 0.27-0.35 for this came from the strain-0.12
scan. Here it is redone at strain
1.0, where s / ln(3.162) IS d ln mu / d ln I.

Usage:  python3 analyze_separability.py
"""
import numpy as np

import nfit

ZMIN = 3.0
LNR = np.log(3.162)                      # ln(I_B / I_A)
MUS = [0.0, 0.15, 0.3, 1.0]

ARMS = [
    ('A  gdot=1.0e-3', 'sweep_steady2d/log.mu*',
     r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', 3.162e-4),
    ('B  gdot=3.162e-3', 'sweep_iscan2/log.g3.162e-3_mu*',
     r'log\.g(?P<g>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
     1.0e-3),
]


def load(pattern, rx):
    out = {}
    for key, runs in nfit.load_sweep(pattern, rx).items():
        out.setdefault(float(dict(key)['mu']), []).extend(runs)
    return out


def gated(runs):
    """Surviving setpoints that are also jammed."""
    Ts, byT = nfit.surviving_setpoints(runs, z_min=ZMIN)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if not z or np.mean(z) >= ZMIN:
            keep.append(t)
    return keep, byT


def jk(A, y, groups, idxs):
    """Calibrated delete-one-group jackknife on the listed coefficients.

    Returns (coef, err) arrays. err is the EXCESS over the analytic null,
    combined with the OLS stat error -- identical policy to fit_local_sys.
    """
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ c
    dof = max(len(y) - A.shape[1], 1)
    cov = float(np.sum(r ** 2) / dof) * np.linalg.inv(A.T @ A)
    keys = sorted(set(groups))
    k = len(keys)
    reps = []
    for g in keys:
        m = groups != g
        if m.sum() <= A.shape[1]:
            continue
        reps.append(np.linalg.lstsq(A[m], y[m], rcond=None)[0])
    reps = np.array(reps)
    out = []
    for i in idxs:
        stat = float(np.sqrt(cov[i, i]))
        if len(reps) >= 3:
            raw = float(np.sqrt((k - 1) / k * np.sum((reps[:, i] - c[i]) ** 2)))
            f = nfit._jk_null_factor_A(A, groups, i)
            null = f * stat if np.isfinite(f) else np.nan
            sysd = (float(np.sqrt(max(raw ** 2 - null ** 2, 0.0)))
                    if np.isfinite(null) else raw)
        else:
            sysd = 0.0
        out.append((float(c[i]), float(np.hypot(stat, sysd))))
    return out


def main():
    A_ = load(*ARMS[0][1:3])
    B_ = load(*ARMS[1][1:3])

    print('=' * 78)
    print('SEPARABILITY OF Theta AND I   --   is  mu * Theta^n = F(I)  the right FORM?')
    print('=' * 78)
    print(f'  arm A  I = {ARMS[0][3]:.3e}   {len(A_)} frictions in sweep_steady2d')
    print(f'  arm B  I = {ARMS[1][3]:.3e}   {len(B_)} frictions in sweep_iscan2 '
          '(fast arm, complete)')
    print(f'  ln(I_B/I_A) = {LNR:.4f}\n')

    print(f'  {"mu_g":>6} {"n(A)":>16} {"n(B)":>16} {"Delta n":>16} {"sigma":>7}')
    print('  ' + '-' * 74)
    rows = []
    for mg in MUS:
        if mg not in A_ or mg not in B_:
            print(f'  {mg:>6} -- missing arm')
            continue
        got = nfit.common_theta0([A_[mg], B_[mg]])
        if got is None:
            print(f'  {mg:>6} -- no common Theta range')
            continue
        T0, (lo, hi) = got

        X, Y, G, JJ = [], [], [], []
        per_arm = {}
        for tag, arm, j in (('A', A_[mg], 0.0), ('B', B_[mg], 1.0)):
            ts, byT = gated(arm)
            sel = [r for t in ts for r in byT[t]
                   if lo * 0.999 <= r['Theta'] <= hi * 1.001
                   and r['Theta'] > 0 and r['mu'] > 0]
            per_arm[tag] = len({round(r['Tgran'], 12) for r in sel})
            for r in sel:
                X.append(np.log(r['Theta'] / T0)); Y.append(np.log(r['mu']))
                G.append(f'{tag}{round(r["Tgran"], 12)}'); JJ.append(j)
        u = np.array(X); y = np.array(Y); J = np.array(JJ)
        groups = np.array(G)
        if per_arm.get('A', 0) < 4 or per_arm.get('B', 0) < 4:
            print(f'  {mg:>6} -- too few setpoints ({per_arm})')
            continue

        # a  s  b  beta  c  gamma
        D = np.vstack([np.ones_like(u), J, u, J * u, u ** 2, J * u ** 2]).T
        (s, se), (beta, bee), (gam, gae) = jk(D, y, groups, [1, 3, 5])

        # per-arm n for display
        nA = nfit.fit_local_sys(A_[mg], T0, restrict_to=gated(A_[mg])[0],
                                z_min=ZMIN)
        nB = nfit.fit_local_sys(B_[mg], T0, restrict_to=gated(B_[mg])[0],
                                z_min=ZMIN)
        dn, dne = -beta, bee
        print(f'  {mg:>6} {nA["n"]:>9.4f}+/-{nA["tot"]:.4f}'
              f' {nB["n"]:>9.4f}+/-{nB["tot"]:.4f}'
              f' {dn:>9.4f}+/-{dne:.4f} {abs(dn)/dne:>6.1f}')
        rows.append(dict(mg=mg, T0=T0, s=s, se=se, dn=dn, dne=dne,
                         gam=gam, gae=gae, nA=nA['n'], nAe=nA['tot'],
                         nB=nB['n'], nBe=nB['tot'], k=per_arm))

    if not rows:
        return

    # joint verdict on parallelism
    print()
    z = np.array([r['dn'] / r['dne'] for r in rows])
    chi2 = float((z ** 2).sum())
    w = np.array([1 / r['dne'] ** 2 for r in rows])
    dnbar = float((w * np.array([r['dn'] for r in rows])).sum() / w.sum())
    dnerr = float(1 / np.sqrt(w.sum()))
    print(f'  pooled Delta n = {dnbar:+.4f} +/- {dnerr:.4f} '
          f'({abs(dnbar)/dnerr:.1f} sigma)     '
          f'chi2 = {chi2:.1f} on {len(rows)} cells')
    print('  -> separability requires Delta n = 0 at EVERY friction.\n')

    print(f'  {"mu_g":>6} {"Delta curvature":>20} {"sigma":>7}')
    print('  ' + '-' * 36)
    for r in rows:
        dc, dce = -2 * r['gam'], 2 * r['gae']
        print(f'  {r["mg"]:>6} {dc:>12.4f}+/-{dce:.4f} {abs(dc)/dce:>6.1f}')

    # the shape of F, and the R-group prediction
    print()
    print('=' * 78)
    print('IF F EXISTS, WHAT IS IT?   and does the R = v_T/(gdot d) group work?')
    print('=' * 78)
    print(f'  {"mu_g":>6} {"dlnmu/dlnI":>18} {"2n (predicted)":>18} {"ratio":>9}')
    print('  ' + '-' * 56)
    for r in rows:
        d, de = r['s'] / LNR, r['se'] / LNR
        pred, prede = 2 * r['nA'], 2 * r['nAe']
        ratio = d / pred if pred else np.nan
        print(f'  {r["mg"]:>6} {d:>11.4f}+/-{de:.4f}'
              f' {pred:>11.4f}+/-{prede:.4f} {ratio:>9.2f}')
    print('\n  ratio = 1 would mean Theta and I collapse onto R ~ sqrt(Theta)/I.')

    # A weaker, more general single-variable hypothesis: mu = H(Theta / I^a,
    # mu_g) for SOME constant a -- R is just the case a = 2. Then
    # d ln mu/d ln I = a * n, so a is measurable per friction and must not move.
    print()
    print(f'  the general one-variable form mu = H(Theta/I^a):  a = '
          '(dlnmu/dlnI)/n must be constant')
    av, ae = [], []
    for r in rows:
        a = (r['s'] / LNR) / r['nA']
        aerr = abs(a) * np.hypot(r['se'] / r['s'], r['nAe'] / r['nA'])
        av.append(a); ae.append(aerr)
        print(f'  {r["mg"]:>6}   a = {a:>6.2f} +/- {aerr:.2f}')
    av, ae = np.array(av), np.array(ae)
    w = 1 / ae ** 2
    ab = float((w * av).sum() / w.sum())
    c2 = float((((av - ab) / ae) ** 2).sum())
    print(f'  common a = {ab:.2f}, chi2/dof = {c2/max(len(av)-1,1):.1f} '
          f'-> {"consistent" if c2/max(len(av)-1,1) < 2 else "REJECTED"}')

    # if n is not constant in I, HOW does it move?
    # Delta n / n is far more nearly constant across friction than Delta n is.
    # If n(I) = n_0(mu_g) * (I/I_0)^k with a SINGLE k, the failure of
    # separability is itself a one-parameter law, and the object that is a
    # function of mu_g alone is n * I^-k rather than n.
    print()
    print('=' * 78)
    print('THE FAILURE HAS A SHAPE:   n(I) = n_0(mu_g) * I^k  with a common k?')
    print('=' * 78)
    ks, kes = [], []
    print(f'  {"mu_g":>6} {"n_B/n_A":>10} {"k":>18}')
    print('  ' + '-' * 38)
    for r in rows:
        nA, nAe, nB, nBe = r['nA'], r['nAe'], r['nB'], r['nBe']
        if nA <= 0 or nB <= 0:
            print(f'  {r["mg"]:>6}   n <= 0, skipped')
            continue
        ratio = nB / nA
        sln = np.hypot(nAe / nA, nBe / nB)
        k, ke = np.log(ratio) / LNR, sln / LNR
        ks.append(k); kes.append(ke)
        print(f'  {r["mg"]:>6} {ratio:>10.4f} {k:>11.4f}+/-{ke:.4f}')
    if len(ks) >= 2:
        ks = np.array(ks); kes = np.array(kes)
        w = 1 / kes ** 2
        kb = float((w * ks).sum() / w.sum()); kbe = float(1 / np.sqrt(w.sum()))
        c2 = float((((ks - kb) / kes) ** 2).sum())
        dof = len(ks) - 1
        print(f'\n  common k = {kb:+.4f} +/- {kbe:.4f}   ({abs(kb)/kbe:.1f} sigma '
              f'from 0)   chi2/dof = {c2/dof:.2f} on {dof} dof')
        print(f'  -> a SINGLE exponent k describes how n shifts with I at every '
              f'friction.\n     n * I^{-kb:+.3f} is then the friction-only object, '
              'in place of n.')
        print('  CAVEAT: two arms give one k per friction exactly, so this tests '
              'only\n  consistency, not the power-law form. The slow arm '
              '(66 runs left) adds the\n  third point that tests the form itself.')


if __name__ == '__main__':
    main()
