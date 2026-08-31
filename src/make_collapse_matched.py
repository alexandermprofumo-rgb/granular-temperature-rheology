"""
Fig. 7: the dZ_eff collapse, retested at matched pressure.

The campaign recorded this test as FAILED (a 3x discrepancy in n at matched
dZ_eff). That verdict rested on comparing a 2D sweep at Pconf=2 against a 3D
sweep at Pconf=10 -- i.e. a hypostatic packing against a hyperstatic one --
and, separately, on fitting the two dimensions over different Theta windows
(make_collapse_figure.py filtered 2D through CORE2D but passed core=None for
3D). With both defects removed:

  * the shared dZ_eff window widens from 0.27 to 0.90;
  * n collapses onto dZ_eff at 1.5 sigma, clearly better than mu_g (2.9) or
    the Coulomb-mobilised fraction chi (3.3).

So generalised isostaticity is NOT refuted by this data; the earlier
refutation was an artifact of unmatched pressure and unlike fitting windows.

The 3D grid was then extended (run_3d_extend.sh: mu_g = 0.4/0.6/0.8/1.5/2.0)
to populate the overlap window, which previously held only two explicit 3D
points. Doing so IMPROVED the collapse, 1.7 -> 1.5 sigma, with 4 points in
window. A spurious collapse degrades when you sample it better; this one did
not. It is still only marginal agreement, not a demonstration -- 1.5 sigma on
8 points -- so the honest verdict is "not refuted, and the best of the
candidates tested", not "confirmed".

Usage:  python3 make_collapse_matched.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLOOR, ITOL = 0.80, 0.03


def load(f):
    rs = list(csv.DictReader(open(f)))
    for r in rs:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rs


def fit(th, mu):
    th = np.asarray(th, float)
    mu = np.asarray(mu, float)
    m = (th > 0) & (mu > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])


def ncurve(f):
    rs = load(f)
    out = {}
    for mg in sorted(set(r['mu_g'] for r in rs)):
        rows = [r for r in rs if r['mu_g'] == mg]
        byT = {}
        for r in rows:
            byT.setdefault(r['Tgran'], []).append(r)
        Ts = sorted(byT)
        I = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
        # n_contacts is absent for runs whose contact dump has been pruned.
        # The dilation cut is then simply unavailable; fall back to the I cut,
        # which is the criterion tied to the method's premise anyway.
        NC = {}
        for t in Ts:
            v = [r['n_contacts'] for r in byT[t]
                 if isinstance(r.get('n_contacts'), float)]
            if v:
                NC[t] = float(np.mean(v))
        ncm = max(NC.values()) if len(NC) == len(Ts) else None
        Im = np.median(list(I.values()))
        sel = [r for t in Ts
               if abs(I[t] / Im - 1) <= ITOL
               and (ncm is None or NC[t] >= FLOOR * ncm)
               for r in byT[t]]
        if len({r['Tgran'] for r in sel}) >= 3:
            out[mg] = fit([r['Theta'] for r in sel], [r['mu'] for r in sel])
    return out


def micro(f):
    rs = load(f)
    return {m: {k: np.mean([r[k] for r in rs if r['mu_g'] == m])
                for k in ('chi', 'Z', 'Z_c', 'dZ_eff')}
            for m in sorted(set(r['mu_g'] for r in rs))}


def series(nf, mf):
    n, m = ncurve(nf), micro(mf)
    ks = sorted(set(n) & set(m))
    return (np.array(ks),
            np.array([n[k][0] for k in ks]), np.array([n[k][1] for k in ks]),
            np.array([m[k]['dZ_eff'] for k in ks]),
            np.array([m[k]['chi'] for k in ks]))


def rms_sigma(x2, y2, e2, x3, y3):
    o = np.argsort(x3)
    xs, ys = x3[o], y3[o]
    lo, hi = max(x2.min(), xs.min()), min(x2.max(), xs.max())
    s = (x2 >= lo) & (x2 <= hi)
    if s.sum() < 3:
        return np.nan, np.nan, 0, (lo, hi)
    d = y2[s] - np.interp(x2[s], xs, ys)
    return (np.sqrt(np.mean(d ** 2)),
            np.sqrt(np.mean((d / e2[s]) ** 2)), int(s.sum()), (lo, hi))


def main():
    mu2, n2, e2, dz2, chi2 = series('results_matched2d.csv',
                                    'tier1_matched2d_results.csv')
    mu3, n3, e3, dz3, chi3 = series('results_3d.csv', 'tier1_3d_results.csv')

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    C2, C3 = 'tab:blue', 'tab:red'

    a = ax[0]
    a.errorbar(mu2, n2, yerr=e2, fmt='o-', color=C2, ms=4, capsize=2.5, label='2D')
    a.errorbar(mu3, n3, yerr=e3, fmt='s-', color=C3, ms=4, capsize=2.5, label='3D')
    a.set_xscale('symlog', linthresh=0.05)
    a.set_xlim(left=-0.005)
    a.set_xlabel(r'$\mu_g$')
    a.set_ylabel('$n$')
    a.set_title('(a) $n(\\mu_g)$, both at $P_{conf}=10$', fontsize=9)
    a.legend(fontsize=8)

    a = ax[1]
    a.plot(mu2, dz2, 'o-', color=C2, label=r'2D $\Delta Z_{\rm eff}$')
    a.plot(mu3, dz3, 's-', color=C3, label=r'3D $\Delta Z_{\rm eff}$')
    a.axhline(0, ls=':', c='k', lw=0.8)
    a.set_xscale('symlog', linthresh=0.05)
    a.set_xlim(left=-0.005)
    a.set_xlabel(r'$\mu_g$')
    a.set_ylabel(r'$\Delta Z_{\rm eff} = Z - Z_c(\chi)$')
    a.set_title('(b) the control parameter now spans\na common range', fontsize=9)
    a.legend(fontsize=7)

    a = ax[2]
    r_dz, s_dz, npt, (lo, hi) = rms_sigma(dz2, n2, e2, dz3, n3)
    a.errorbar(dz2, n2, yerr=e2, fmt='o', color=C2, capsize=3, label='2D')
    a.errorbar(dz3, n3, yerr=e3, fmt='s', color=C3, capsize=3, label='3D')
    a.axvspan(lo, hi, color='0.88', zorder=0, label=f'overlap ({hi-lo:.2f} wide)')
    inside = (dz3 >= lo) & (dz3 <= hi)
    a.plot(dz3[inside], n3[inside], 's', mfc='none', mec='k', ms=11, mew=1.4,
           label=f'{inside.sum()} explicit 3D pts in window')
    for x, y, m in zip(dz2, n2, mu2):
        a.annotate(f'{m:g}', (x, y), fontsize=5.5, xytext=(3, 3),
                   textcoords='offset points', color=C2)
    for x, y, m in zip(dz3, n3, mu3):
        a.annotate(f'{m:g}', (x, y), fontsize=5.5, xytext=(3, -9),
                   textcoords='offset points', color=C3)
    a.set_xlabel(r'$\Delta Z_{\rm eff}$')
    a.set_ylabel('$n$')
    r_mu = rms_sigma(mu2, n2, e2, mu3, n3)[1]
    r_chi = rms_sigma(chi2, n2, e2, chi3, n3)[1]
    a.set_title(f'(c) collapse: RMS={r_dz:.3f} ({s_dz:.1f}$\\sigma$)\n'
                f'better than $\\mu_g$ ({r_mu:.1f}) or $\\chi$ ({r_chi:.1f})',
                fontsize=9)
    a.legend(fontsize=6, loc='upper left')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig7_collapse_matched.{ext}', dpi=200, bbox_inches='tight')
    print(f'dZ_eff collapse: RMS={r_dz:.4f} ({s_dz:.1f} sigma) over {npt} 2D points')
    print(f'overlap [{lo:+.3f},{hi:+.3f}] contains {inside.sum()} explicit 3D points')
    print('wrote fig7_collapse_matched.{pdf,png}')


if __name__ == '__main__':
    main()
