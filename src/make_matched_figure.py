"""
Fig. 6: what changes when the two dimensions are compared at the SAME pressure.

(a) Z(Pconf) in 2D. The production sweep ran at Pconf=2, where the packing is
    hypostatic (Z=3.30 < 4). At Pconf=10 it sits at Z=4.04, its frictionless
    isostatic value -- which is where the 3D sweep was already calibrated
    (Z~6 at Pconf=10). Both dimensions are isostatic at the same pressure.

(b) n(mu_g) for the three sweeps under one consistent setpoint criterion. At
    matched pressure the two dimensions share a frictionless exponent
    (0.061 vs 0.062) and a peak location (mu_g ~ 0.15-0.2). At Pconf=2 the 2D
    curve peaks elsewhere and sits well above.

(c) robustness: fitted n(0) and peak position as the dilation cut is varied.
    Both Pconf=10 sweeps are flat; the Pconf=2 sweep is not, because a
    hypostatic packing dilates across the Theta scan and the fit window then
    selects different physical states.

Usage:  python3 make_matched_figure.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from in.calib_Z_2d, mu_g = 0
CALIB_P = [2.0, 5.0, 10.0, 20.0]
CALIB_Z = [3.297, 3.800, 4.036, 4.215]

SW = [('results.csv', '2D  $P_{conf}=2$ (production)', 'tab:green', 'o'),
      ('results_matched2d.csv', '2D  $P_{conf}=10$ (matched)', 'tab:blue', 'o'),
      ('results_3d.csv', '3D  $P_{conf}=10$', 'tab:red', 's')]


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
    m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])


def curve(f, floor=0.80, itol=0.03):
    rs = load(f)
    out = {}
    for mg in sorted(set(r['mu_g'] for r in rs)):
        rows = [r for r in rs if r['mu_g'] == mg]
        byT = {}
        for r in rows:
            byT.setdefault(r['Tgran'], []).append(r)
        Ts = sorted(byT)
        I = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
        NC = {t: np.mean([r['n_contacts'] for r in byT[t]]) for t in Ts}
        Im, ncm = np.median(list(I.values())), max(NC.values())
        sel = [r for t in Ts
               if abs(I[t] / Im - 1) <= itol and NC[t] >= floor * ncm
               for r in byT[t]]
        if len({r['Tgran'] for r in sel}) >= 3:
            out[mg] = fit([r['Theta'] for r in sel], [r['mu'] for r in sel])
    return out


def main():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

    a = ax[0]
    a.plot(CALIB_P, CALIB_Z, 'ko-')
    a.axhline(4.0, ls='--', c='0.5', lw=1)
    a.annotate('2D frictionless isostatic $Z=4$', (2.2, 4.06), fontsize=7.5, color='0.35')
    for P, Z, c in [(2.0, 3.297, 'tab:green'), (10.0, 4.036, 'tab:blue')]:
        a.plot([P], [Z], 'o', ms=11, mfc='none', mec=c, mew=2)
    a.annotate('production\n(hypostatic)', (2.0, 3.297), fontsize=7.5,
               xytext=(3.0, 3.36), color='tab:green')
    a.annotate('matched\n(isostatic)', (10.0, 4.036), fontsize=7.5,
               xytext=(7.0, 3.72), color='tab:blue')
    a.set_xscale('log')
    a.set_xticks(CALIB_P)
    a.set_xticklabels([f'{p:g}' for p in CALIB_P])
    a.minorticks_off()
    a.set_xlabel(r'$P_{conf}$')
    a.set_ylabel(r'coordination $Z$  ($\mu_g=0$)')
    a.set_title('(a) pressure, not dimension, sets $Z$', fontsize=9)

    a = ax[1]
    for f, lab, c, mk in SW:
        d = curve(f)
        x = np.array(sorted(d))
        y = np.array([d[k][0] for k in x])
        e = np.array([d[k][1] for k in x])
        a.errorbar(x, y, yerr=e, fmt=mk + '-', color=c, ms=4, lw=1.2,
                   capsize=2.5, label=lab)
    a.axhline(0.25, ls=':', c='tab:blue', lw=1)
    a.axhline(1 / 6, ls=':', c='tab:red', lw=1)
    a.annotate('theory 2D $=1/4$', (0.0012, 0.253), fontsize=7, color='tab:blue')
    a.annotate('theory 3D $=1/6$', (0.0012, 0.170), fontsize=7, color='tab:red')
    a.set_xscale('symlog', linthresh=1e-3)
    a.set_xlim(left=0)
    a.set_xlabel(r'$\mu_g$')
    a.set_ylabel('$n$')
    a.set_title('(b) at matched $P$ the dimensions agree at $\\mu_g=0$\n'
                'and peak together', fontsize=9)
    a.legend(fontsize=6.5, loc='lower right')

    a = ax[2]
    floors = [0.0, 0.70, 0.80, 0.85, 0.90]
    for f, lab, c, mk in SW:
        n0, pk = [], []
        for fl in floors:
            d = curve(f, floor=fl)
            n0.append(d.get(0.0, (np.nan, 0))[0])
            pk.append(max(d, key=lambda k: d[k][0]) if d else np.nan)
        a.plot(floors, n0, mk + '-', color=c, ms=4, label=lab)
    a.set_xlabel('dilation cut (min contacts / max)')
    a.set_ylabel('$n(\\mu_g=0)$')
    a.set_title('(c) robustness: $P_{conf}=10$ is flat,\n'
                '$P_{conf}=2$ is not', fontsize=9)
    a.legend(fontsize=6.5)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig6_matched.{ext}', dpi=200, bbox_inches='tight')
    print('wrote fig6_matched.{pdf,png}')


if __name__ == '__main__':
    main()
