"""
Second figure: what n actually does.

  (a) n(mu_g) in 2D and 3D with total errors (stat + window jackknife), the 3D
      points carrying the convergence correction of section 6a. The geometric
      predictions 1/4 and 1/6 are marked -- the frictionless baselines sit far
      below both, which is the paper's central claim.
  (b) n against the inertial number at four frictions. The frictionless point is
      flat; the dependence switches on with friction.

Panel (a) is the constraint list made visible; panel (b) is the reason every
value in it needs a stated I.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MU = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
N2 = [0.0816, 0.1240, 0.1669, 0.2187, 0.2113, 0.2273, 0.1442, 0.0745]
E2 = [0.0247, 0.0242, 0.0152, 0.0202, 0.0153, 0.0180, 0.0110, 0.0135]
N3 = [0.0445, 0.1119, 0.1526, 0.1689, 0.1861, 0.1837, 0.1706, 0.1209]
E3 = [0.0236, 0.0126, 0.0123, 0.0182, 0.0252, 0.0338, 0.0363, 0.0398]

IV = [1.0013e-4, 3.1719e-4, 1.0e-3]
NI = {0.0:  ([0.0616, 0.0097, 0.0456], [0.0151, 0.0260, 0.0342]),
      0.15: ([0.1960, 0.2015, 0.1691], [0.0177, 0.0173, 0.0131]),
      0.3:  ([0.2235, 0.1919, 0.1176], [0.0252, 0.0122, 0.0092]),
      1.0:  ([0.1791, 0.0931, 0.0874], [0.0215, 0.0179, 0.0249])}

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))

a = ax[0]
a.errorbar(MU, N2, yerr=E2, fmt='o-', color='#4C72B0', capsize=3, lw=1.7, ms=5, label='2D')
a.errorbar(MU, N3, yerr=E3, fmt='s--', color='#C44E52', capsize=3, lw=1.7, ms=5,
           label='3D (convergence-corrected)')
a.axhline(0.25, color='#4C72B0', ls=':', lw=1.4)
a.axhline(1/6, color='#C44E52', ls=':', lw=1.4)
a.text(0.62, 0.253, r'geometric  $n=1/4$ (2D)', fontsize=7.2, color='#4C72B0')
a.text(0.62, 0.170, r'geometric  $n=1/6$ (3D)', fontsize=7.2, color='#C44E52')
a.annotate('frictionless baselines:\n6.8$\\sigma$ / 5.2$\\sigma$ below theory',
           xy=(0.0, 0.063), xytext=(0.11, 0.012), fontsize=7.2,
           arrowprops=dict(arrowstyle='->', lw=.9, color='0.3'), color='0.25')
a.set_xlabel(r'$\mu_g$'); a.set_ylabel(r'$n(\Theta_0)$')
a.set_title('(a)  n against friction, both dimensions', fontsize=10, loc='left')
a.legend(fontsize=8, loc='lower right'); a.set_ylim(-0.01, 0.29)

b = ax[1]
for mg, col in zip([0.0, 0.15, 0.3, 1.0], ['#8C8C8C', '#55A868', '#4C72B0', '#C44E52']):
    y, e = NI[mg]
    b.errorbar(IV, y, yerr=e, fmt='o-', color=col, capsize=3, lw=1.6, ms=5,
               label=r'$\mu_g=%g$' % mg)
b.set_xscale('log'); b.set_xlabel('inertial number  $I$'); b.set_ylabel(r'$n(\Theta_0)$')
b.set_title('(b)  n depends on I — but not at zero friction\n'
            r'$\partial n/\partial\ln I = -0.032\pm0.006$ (5.7$\sigma$) overall',
            fontsize=10, loc='left')
b.legend(fontsize=8); b.set_ylim(-0.01, 0.29)

for x in ax:
    x.grid(alpha=.25, lw=.5)
fig.tight_layout(); fig.savefig('fig_constraints.png', dpi=190)
print('wrote fig_constraints.png')
