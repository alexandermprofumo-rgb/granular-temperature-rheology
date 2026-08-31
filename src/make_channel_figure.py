"""
The paper's central figure: the channel decomposition of n.

Three panels:
  (a) n(mu_g) in 2D decomposed into C_c + C_n + C_t (stacked), 15 frictions
  (b) channel SHARES against friction, 2D and 3D, with a_t MEASURED
  (c) the identity check -- mu against (a_c+a_n+a_t)/2 over all 432 runs

Panel (b) is the result: the dominant channel inverts, in both dimensions,
independently. It uses the measured-a_t values (sweep_tracking_forces,
sweep_tracking_forces_3d) rather than the closure-derived ones, because closure
biases the tangential share upward at high friction -- which is exactly where
the claim lives.

Usage:  python3 make_channel_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# panel (a): 2D stacked channels, from analyze_rb_fine (closure a_t)
MU2 = [0, .05, .06, .08, .1, .11, .12, .13, .15, .17, .2, .25, .3, .5, 1.]
CC2 = [.0036, .0352, .0087, .0369, .0218, .0510, .0455, .0558, .0516, .0710,
       .0555, .0532, .0619, .0331, .0273]
CN2 = [.0749, .0881, .0970, .0949, .1385, .1384, .1471, .1484, .1517, .1542,
       .1203, .1173, .1172, .0615, .0020]
CT2 = [.0045, .0025, .0042, .0078, .0112, .0085, .0124, .0125, .0142, .0211,
       .0321, .0312, .0381, .0418, .0436]

# panel (b): SHARES with a_t measured
M2 = [0.03, 0.1, 0.3, 1.0, 2.0]
S2 = np.array([[-.01, 1.00, .01], [-.02, .97, .05], [.19, .69, .12],
               [.14, .53, .33], [.12, .38, .50]])
M3 = [0.05, 0.3, 0.5, 1.0, 2.0]
S3 = np.array([[.34, .50, .16], [.19, .38, .43], [.19, .24, .58],
               [.12, .19, .70], [.18, .23, .59]])

COL = {'c': '#4C72B0', 'n': '#C44E52', 't': '#55A868'}
LAB = {'c': r'$C_c$  fabric', 'n': r'$C_n$  normal force', 't': r'$C_t$  tangential'}


def main():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # (a) stacked decomposition
    a = ax[0]
    a.stackplot(MU2, CC2, CN2, CT2,
                colors=[COL['c'], COL['n'], COL['t']],
                labels=[LAB['c'], LAB['n'], LAB['t']], alpha=.85)
    a.plot(MU2, np.array(CC2) + np.array(CN2) + np.array(CT2), 'k-', lw=1.6,
           label=r'$n = C_c+C_n+C_t$')
    # a_t here is CLOSURE-derived (these dumps carry no tangential force), and
    # closure biases C_t upward at high friction -- so the green band beyond
    # mu_g ~ 0.3 is an upper bound. Panel (b) uses measured a_t and is the one
    # to read for the high-friction shares. Shading marks where they diverge.
    a.axvspan(0.3, 1.0, color='0.5', alpha=.13, zorder=0)
    a.text(0.44, 0.238, '$a_t$ by closure:\n$C_t$ is an upper\nbound here',
           fontsize=6.8, color='0.25', va='top')
    a.set_xscale('log'); a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'$n(\Theta_0)$')
    a.set_title('(a)  2D: n decomposed by channel', fontsize=10, loc='left')
    a.legend(fontsize=7.5, loc='upper left', framealpha=.9)
    a.set_xlim(0.04, 1.0)

    # (b) shares, both dimensions, measured a_t
    b = ax[1]
    for k, j in (('c', 0), ('n', 1), ('t', 2)):
        b.plot(M2, S2[:, j], 'o-', color=COL[k], lw=1.8, ms=5, label=LAB[k])
        b.plot(M3, S3[:, j], 's--', color=COL[k], lw=1.4, ms=5, alpha=.75)
    b.axhline(0, color='k', lw=.6)
    b.set_xscale('log'); b.set_xlabel(r'$\mu_g$')
    b.set_ylabel('share of $n$')
    b.set_title('(b)  the dominant channel inverts\n'
                'solid 2D, dashed 3D  ($a_t$ measured)', fontsize=10, loc='left')
    b.legend(fontsize=7.5, loc='center left', framealpha=.9)
    b.set_ylim(-.15, 1.1)

    # (c) identity check
    c = ax[2]
    try:
        import csv
        rows = list(csv.DictReader(open('rb_channels.csv')))
        mu = np.array([float(r['mu']) for r in rows])
        s = np.array([(float(r['a_c']) + float(r['a_n']) + float(r['a_t'])) / 2
                      for r in rows])
        mg = np.array([float(r['mu_g']) for r in rows])
        sc = c.scatter(mu, s, c=np.log10(mg), s=7, cmap='viridis', alpha=.7)
        plt.colorbar(sc, ax=c, label=r'$\log_{10}\mu_g$')
        lim = [min(mu.min(), s.min()) * .95, max(mu.max(), s.max()) * 1.05]
        c.plot(lim, lim, 'k--', lw=1)
        c.set_xlim(lim); c.set_ylim(lim)
        r = s / mu
        c.set_title(f'(c)  $\\mu=(a_c+a_n+a_t)/2$\nratio {r.mean():.4f} '
                    f'$\\pm$ {r.std():.4f}, n={len(r)}', fontsize=10, loc='left')
    except FileNotFoundError:
        c.text(.5, .5, 'rb_channels.csv not found', ha='center')
    c.set_xlabel(r'$\mu$ measured'); c.set_ylabel(r'$(a_c+a_n+a_t)/2$')

    for x in ax:
        x.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig('fig_channels.png', dpi=190)
    print('wrote fig_channels.png')


if __name__ == '__main__':
    main()
