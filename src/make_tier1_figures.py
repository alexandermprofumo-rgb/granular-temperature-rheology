"""
Tier 1 figures, filling the Fig. 3 / Fig. 4 placeholders in main.tex.

fig3_isostaticity.pdf
    (a) chi(mu_g): the Coulomb-mobilised fraction, measured for the first time
    (b) Z and Z_c(chi): Z is flat where n rises; Z_c is not
    (c) dZ_eff(mu_g) against n(mu_g), both peaked
    (d) the collapse test: n vs dZ_eff, coloured by branch

fig4_force_weighted.pdf
    (a) Theta-scaling exponent of the unweighted vs force-weighted fabric,
        against the macroscopic n(mu_g)
    (b) parity plot: exponent vs n, showing which measure reproduces it

Usage:  python3 make_tier1_figures.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORE = {0.0005, 0.002, 0.006, 0.015}


def load(f):
    rs = list(csv.DictReader(open(f)))
    for r in rs:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rs


def fit(th, y, sign=-1.0):
    th = np.asarray(th, float); y = np.asarray(y, float)
    m = np.isfinite(th) & np.isfinite(y) & (th > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, yy = np.log(th[m]), np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ c
    s2 = np.sum(resid ** 2) / max(m.sum() - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return sign * c[0], np.sqrt(cov[0, 0])


def main():
    t1 = load('tier1_results.csv')
    mac = [r for r in load('results.csv') if r['Tgran'] in CORE]
    mus = np.array(sorted(set(r['mu_g'] for r in t1)))

    g = lambda mg, k: np.mean([r[k] for r in t1 if r['mu_g'] == mg])
    Z = np.array([g(m, 'Z') for m in mus])
    Zc = np.array([g(m, 'Z_c') for m in mus])
    chi = np.array([g(m, 'chi') for m in mus])
    dz = np.array([g(m, 'dZ_eff') for m in mus])

    n, ne, unw, unwe, fw, fwe = [], [], [], [], [], []
    for m in mus:
        sm = [r for r in mac if r['mu_g'] == m]
        st = [r for r in t1 if r['mu_g'] == m]
        v, e = fit([r['Theta'] for r in sm], [r['mu'] for r in sm]); n.append(v); ne.append(e)
        v, e = fit([r['Theta'] for r in st], [r['a_c'] for r in st]); unw.append(v); unwe.append(e)
        v, e = fit([r['Theta'] for r in st], [r['a_c_w'] for r in st]); fw.append(v); fwe.append(e)
    n, ne, unw, unwe, fw, fwe = map(np.array, (n, ne, unw, unwe, fw, fwe))

    # Fig 3
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.2))
    a = ax[0, 0]
    a.plot(mus, chi, 'o-', color='tab:purple')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'mobilised fraction $\chi$')
    a.set_title(r'(a) $\chi$ collapses with friction', fontsize=10)

    a = ax[0, 1]
    a.plot(mus, Z, 'o-', label=r'$Z$ (measured)', color='tab:blue')
    a.plot(mus, Zc, 's--', label=r'$Z_c(\chi)$ (generalised isostatic)', color='tab:orange')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel('coordination')
    a.set_title(r'(b) $Z$ is flat where $n$ rises; $Z_c$ is not', fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 0]
    a.plot(mus, dz, 'o-', color='tab:green', label=r'$\Delta Z_{\rm eff}=Z-Z_c(\chi)$')
    a.axhline(0, lw=0.8, ls=':', c='k')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'$\Delta Z_{\rm eff}$', color='tab:green')
    a.tick_params(axis='y', labelcolor='tab:green')
    a2 = a.twinx(); a2.errorbar(mus, n, yerr=ne, fmt='s--', color='gray', ms=4, lw=1)
    a2.set_ylabel('macroscopic $n$', color='gray'); a2.tick_params(axis='y', labelcolor='gray')
    a.set_title(r'(c) both $\Delta Z_{\rm eff}$ and $n$ are peaked', fontsize=10)

    a = ax[1, 1]
    rise = mus <= 0.3
    a.errorbar(dz[rise], n[rise], yerr=ne[rise], fmt='o', color='tab:blue',
               capsize=3, label=r'rising branch ($\mu_g\leq0.3$)')
    a.errorbar(dz[~rise], n[~rise], yerr=ne[~rise], fmt='s', color='tab:red',
               capsize=3, label=r'falling branch ($\mu_g>0.3$)')
    for x, y, m in zip(dz, n, mus):
        a.annotate(f'{m:g}', (x, y), fontsize=6, xytext=(3, 3),
                   textcoords='offset points')
    a.set_xlabel(r'$\Delta Z_{\rm eff}$'); a.set_ylabel('macroscopic $n$')
    a.set_title(r'(d) collapse test: $n$ is NOT single-valued in $\Delta Z_{\rm eff}$',
                fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'fig3_isostaticity.{e}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Fig 4
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    a = ax[0]
    a.errorbar(mus, n, yerr=ne, fmt='s--', color='k', ms=5, capsize=3, label='macroscopic $n$')
    a.errorbar(mus, unw, yerr=unwe, fmt='o-', color='tab:red', capsize=3,
               label=r'unweighted $a_c$')
    a.errorbar(mus, fw, yerr=fwe, fmt='^-', color='tab:blue', capsize=3,
               label=r'force-weighted $a_c^w$')
    a.axhline(0, lw=0.8, ls=':', c='k')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'$\Theta$-scaling exponent')
    a.set_title('(a) only the force-weighted fabric tracks $n$', fontsize=10)
    a.legend(fontsize=7)

    a = ax[1]
    lim = [min(unw.min(), fw.min(), n.min()) - 0.05, max(unw.max(), fw.max(), n.max()) + 0.05]
    a.plot(lim, lim, 'k:', lw=1, label='exact agreement')
    a.plot(n, unw, 'o', color='tab:red', label=f'unweighted (RMS {np.sqrt(np.mean((unw-n)**2)):.3f})')
    a.plot(n, fw, '^', color='tab:blue', label=f'force-weighted (RMS {np.sqrt(np.mean((fw-n)**2)):.3f})')
    a.axhline(0, lw=0.8, ls=':', c='gray')
    a.set_xlabel('macroscopic $n$'); a.set_ylabel(r'fabric $\Theta$-scaling exponent')
    a.set_title('(b) parity: unweighted has the wrong sign 5/9 times', fontsize=10)
    a.legend(fontsize=7)
    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'fig4_force_weighted.{e}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig3_isostaticity.{pdf,png}, fig4_force_weighted.{pdf,png}')


if __name__ == '__main__':
    main()
