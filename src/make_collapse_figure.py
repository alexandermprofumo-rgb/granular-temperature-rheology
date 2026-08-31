"""
Fig. 5: the generalised-isostaticity collapse test, 2D and 3D together.

This is the figure the campaign called the "money plot". What it actually
shows is that the test cannot be decided as the sweeps stand, and why:

 (a) chi(mu_g) in both dimensions -- the Coulomb-mobilised fraction, measured
     here for the first time. It collapses fast in both, and does so exactly
     where Z is flat, which is what lets Z_c(chi) supply variation that Z
     alone cannot.
 (b) Z and Z_c(chi). 2D sits BELOW its generalised isostatic point
     (hypostatic); 3D sits ABOVE it (hyperstatic).
 (c) dZ_eff and n against mu_g. In 3D the two peak at the same mu_g (0.1) and
     correlate at +0.91; in 2D they peak in different places (+0.57).
 (d) the collapse itself: n vs dZ_eff. The two dimensions occupy nearly
     disjoint ranges -- overlap is only 0.27 wide out of a combined span of
     ~2.1 -- so no joint collapse can be asserted or refuted from this data.

Attempting to manufacture overlap by using confining pressure as an
independent lever on dZ_eff failed for a physical reason: holding I fixed
requires gdot ~ sqrt(P), and the resulting shear heating overwhelms the
thermostat (at Pconf=50 the measured Theta is non-monotonic in the setpoint).
That is recorded here rather than hidden, since it constrains how any future
attempt must be designed.

Usage:  python3 make_collapse_figure.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORE2D = {0.0005, 0.002, 0.006, 0.015}


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
    r = yy - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return sign * c[0], np.sqrt(cov[0, 0])


def series(t1f, macf, core):
    t1 = load(t1f)
    mac = load(macf)
    if core:
        mac = [r for r in mac if r['Tgran'] in core]
    mus = np.array(sorted(set(r['mu_g'] for r in t1)))
    n, ne, dz, Z, Zc, chi = [], [], [], [], [], []
    for m in mus:
        sm = [r for r in mac if r['mu_g'] == m]
        st = [r for r in t1 if r['mu_g'] == m]
        v, e = fit([r['Theta'] for r in sm], [r['mu'] for r in sm])
        n.append(v); ne.append(e)
        dz.append(np.mean([r['dZ_eff'] for r in st]))
        Z.append(np.mean([r['Z'] for r in st]))
        Zc.append(np.mean([r['Z_c'] for r in st]))
        chi.append(np.mean([r['chi'] for r in st]))
    return (mus, np.array(n), np.array(ne), np.array(dz),
            np.array(Z), np.array(Zc), np.array(chi))


def main():
    d2 = series('tier1_results.csv', 'results.csv', CORE2D)
    d3 = series('tier1_3d_results.csv', 'results_3d.csv', None)
    (m2, n2, e2, dz2, Z2, Zc2, chi2) = d2
    (m3, n3, e3, dz3, Z3, Zc3, chi3) = d3

    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.6))
    C2, C3 = 'tab:blue', 'tab:red'

    a = ax[0, 0]
    a.plot(m2, chi2, 'o-', color=C2, label='2D')
    a.plot(m3, chi3, 's-', color=C3, label='3D')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'mobilised fraction $\chi$')
    a.set_title(r'(a) $\chi$ collapses with friction in both dimensions', fontsize=9)
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(m2, Z2, 'o-', color=C2, label=r'2D  $Z$')
    a.plot(m2, Zc2, 'o--', color=C2, mfc='none', label=r'2D  $Z_c(\chi)$')
    a.plot(m3, Z3, 's-', color=C3, label=r'3D  $Z$')
    a.plot(m3, Zc3, 's--', color=C3, mfc='none', label=r'3D  $Z_c(\chi)$')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel('coordination')
    a.set_title('(b) 2D is hypostatic ($Z<Z_c$), 3D hyperstatic', fontsize=9)
    a.legend(fontsize=6, ncol=2)

    a = ax[1, 0]
    a.plot(m2, dz2, 'o-', color=C2, label=r'2D $\Delta Z_{\rm eff}$')
    a.plot(m3, dz3, 's-', color=C3, label=r'3D $\Delta Z_{\rm eff}$')
    a.axhline(0, lw=0.8, ls=':', c='k')
    a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'$\Delta Z_{\rm eff}$')
    a2 = a.twinx()
    a2.errorbar(m2, n2, yerr=e2, fmt='o:', color=C2, ms=3, lw=0.8, alpha=0.45)
    a2.errorbar(m3, n3, yerr=e3, fmt='s:', color=C3, ms=3, lw=0.8, alpha=0.45)
    a2.set_ylabel('$n$ (faded)', fontsize=8)
    a.set_title(r'(c) $\Delta Z_{\rm eff}$ (solid) vs $n$ (faded): 3D peaks'
                '\n' r'together at $\mu_g=0.1$, 2D does not', fontsize=9)
    a.legend(fontsize=7, loc='upper left')

    a = ax[1, 1]
    a.errorbar(dz2, n2, yerr=e2, fmt='o', color=C2, capsize=3, label='2D')
    a.errorbar(dz3, n3, yerr=e3, fmt='s', color=C3, capsize=3, label='3D')
    lo = max(dz2.min(), dz3.min()); hi = min(dz2.max(), dz3.max())
    a.axvspan(lo, hi, color='0.85', zorder=0,
              label=f'overlap ({hi-lo:.2f} wide)')
    for x, y, m in zip(dz2, n2, m2):
        a.annotate(f'{m:g}', (x, y), fontsize=5.5, xytext=(3, 3),
                   textcoords='offset points', color=C2)
    for x, y, m in zip(dz3, n3, m3):
        a.annotate(f'{m:g}', (x, y), fontsize=5.5, xytext=(3, -8),
                   textcoords='offset points', color=C3)
    a.set_xlabel(r'$\Delta Z_{\rm eff} = Z - Z_c(\chi)$')
    a.set_ylabel('fabric-erasure exponent $n$')
    a.set_title('(d) collapse test: the two dimensions barely overlap,\n'
                'so no joint collapse is decidable', fontsize=9)
    a.legend(fontsize=7)

    fig.tight_layout()
    for e in ('pdf', 'png'):
        fig.savefig(f'fig5_collapse.{e}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'2D dZ_eff [{dz2.min():+.3f}, {dz2.max():+.3f}]  '
          f'3D [{dz3.min():+.3f}, {dz3.max():+.3f}]  '
          f'overlap {hi-lo:.3f} of combined span '
          f'{max(dz2.max(),dz3.max())-min(dz2.min(),dz3.min()):.3f}')
    print('wrote fig5_collapse.{pdf,png}')


if __name__ == '__main__':
    main()
