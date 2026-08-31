"""
Generate Figs. 1 and 2 of the paper draft directly from the production
result CSVs, so the [FIG] placeholders in main.tex can be filled.

Fig. 1 (fig1_nofmu.pdf): (a) 2D n(mu_g); (b) 3D n(mu_g); (c) Z(mu_g) both
dimensions; (d) both curves normalized to their own peak.
Fig. 2 (fig2_mechanisms.pdf): the three falsified microscopic mechanisms
plotted against the macroscopic n(mu_g) target.

Conventions matched to the July 2026 report:
  - 2D exponents are fitted in the core Theta window {5e-4, 2e-3, 6e-3,
    1.5e-2}; 3D uses the full tested range.
  - Z = 2 <n_contacts> / N with N = 4000, averaging per-Theta first.
  - sigma_n is the standard error of the pooled log-log fit.

Usage:
    python3 make_figures.py            # writes PDF + PNG next to this file
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_GRAINS = 4000
CORE_WINDOW_2D = {0.0005, 0.002, 0.006, 0.015}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rows


def fit_exponent(sub, y_key='mu', sign=-1.0):
    """Pooled log-log fit; returns (exponent, stderr)."""
    x = np.log([r['Theta'] for r in sub])
    y = np.log([abs(r[y_key]) for r in sub])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(sub) - 2, 1)
    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return sign * coef[0], np.sqrt(cov[0, 0])


def coordination(sub):
    """Z = 2 <n_contacts> / N, averaging within each Theta first."""
    thetas = sorted(set(r['Tgran'] for r in sub))
    per_T = [np.mean([r['n_contacts'] for r in sub if r['Tgran'] == t])
             for t in thetas]
    return 2 * np.mean(per_T) / N_GRAINS


def curve(rows, window=None):
    """Return (mu_g, n, sigma_n, Z) arrays for one dimensionality."""
    if window is not None:
        rows = [r for r in rows if r['Tgran'] in window]
    out = []
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            continue
        n, err = fit_exponent(sub)
        out.append((mg, n, err, coordination(sub)))
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def symlog_axis(ax, linthresh=1e-3):
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_xlabel(r'grain friction $\mu_g$')
    # mu_g < 0 is unphysical; symlog would otherwise pad the axis into it
    ax.set_xlim(left=0)


def make_fig1(base2d, base3d, outstem):
    mu2, n2, e2, Z2 = curve(load(base2d), CORE_WINDOW_2D)
    mu3, n3, e3, Z3 = curve(load(base3d))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    ax = axes[0, 0]
    ax.errorbar(mu2, n2, yerr=e2, marker='o', color='tab:blue', capsize=3, lw=1.5)
    ax.axhline(0.125, ls=':', c='k', lw=1, label=r'$n=1/8$ (observed, frictional)')
    ax.axhline(0.25, ls='--', c='k', lw=1, label=r'$n=1/4$ (mean field, frictionless)')
    ax.set_ylabel('fitted $n$')
    ax.set_title('(a) 2D: rise, overshoot, decline', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    symlog_axis(ax)

    ax = axes[0, 1]
    ax.errorbar(mu3, n3, yerr=e3, marker='s', color='tab:red', capsize=3, lw=1.5)
    ax.axhline(1 / 6, ls=':', c='k', lw=1, label=r'$n=1/6$ (mean field / observed)')
    ax.set_ylabel('fitted $n$')
    ax.set_title('(b) 3D: same shape, sharper and higher', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    symlog_axis(ax, linthresh=1e-2)

    ax = axes[1, 0]
    ax.plot(mu2, Z2, 'o-', color='tab:blue', label='2D')
    ax.plot(mu3, Z3, 's-', color='tab:red', label='3D')
    ax.set_ylabel('coordination number $Z$')
    ax.set_title('(c) $Z$ falls smoothly: explains the\nfalling branch, not the rise',
                 fontsize=10)
    ax.legend(fontsize=8)
    symlog_axis(ax)

    ax = axes[1, 1]
    ax.plot(mu2, n2 / n2.max(), 'o-', color='tab:blue', label='2D (norm. to own peak)')
    ax.plot(mu3, n3 / n3.max(), 's-', color='tab:red', label='3D (norm. to own peak)')
    ax.set_ylabel(r'$n(\mu_g)/n_{\rm peak}$')
    ax.set_title('(d) Shared qualitative shape in\nboth dimensions', fontsize=10)
    ax.legend(fontsize=8)
    symlog_axis(ax)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return (mu2, n2, e2, Z2), (mu3, n3, e3, Z3)


def make_fig2(hypothesis_specs, target_mu, target_n, outstem):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, (title, verdict, path, col, color) in zip(axes, hypothesis_specs):
        rows = load(path)
        mus, vals, errs = [], [], []
        for mg in sorted(set(r['mu_g'] for r in rows)):
            sub = [r for r in rows if r['mu_g'] == mg]
            if len(sub) < 3:
                continue
            v, e = fit_exponent(sub, y_key=col, sign=+1.0)
            mus.append(mg); vals.append(v); errs.append(e)
        ax.errorbar(mus, vals, yerr=errs, marker='o', color=color, capsize=3, lw=1.5)
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_xlabel(r'$\mu_g$')
        ax.set_ylabel(f'{title} scaling exponent', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_title(f'{title}\nTest result: {verdict}', fontsize=9)

        ax2 = ax.twinx()
        ax2.plot(target_mu, target_n, ls='--', color='gray', marker='.', lw=1)
        ax2.set_ylabel('macroscopic $n$ (target, dashed)', color='gray', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='gray')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{outstem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    (mu2, n2, e2, Z2), (mu3, n3, e3, Z3) = make_fig1(
        f'{d}/results.csv', f'{d}/results_3d.csv', f'{d}/fig1_nofmu')

    # Macroscopic target evaluated at the mu_g values the tracking runs used.
    # mu_g = 0.05 is not on the macroscopic grid, so interpolate rather than
    # snapping to a neighbour (which would misalign the two curves in x).
    tracked = np.array([0.0, 0.05, 0.1, 0.3, 1.0])
    tgt_mu, tgt_n = tracked, np.interp(tracked, mu2, n2)

    make_fig2([
        (r'sliding velocity $|v_t|$', 'FLAT (friction has no effect)',
         f'{d}/vt_sweep_results.csv', 'mean_vt', 'tab:blue'),
        ('contact persistence time', 'WRONG SIGN of prediction',
         f'{d}/persistence_results.csv', 'mean_persist', 'tab:green'),
        ('reorientation rate per episode', 'OPPOSITE direction to $n$',
         f'{d}/reorientation_results.csv', 'mean_rate', 'tab:purple'),
    ], tgt_mu, tgt_n, f'{d}/fig2_mechanisms')

    print('Fig. 1  2D:', ' '.join(f'{v:.3f}' for v in n2))
    print('Fig. 1  3D:', ' '.join(f'{v:.3f}' for v in n3))
    print('wrote fig1_nofmu.{pdf,png}, fig2_mechanisms.{pdf,png}')


if __name__ == '__main__':
    main()
