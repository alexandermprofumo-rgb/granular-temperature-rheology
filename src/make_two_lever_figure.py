"""
Fig. 8: why the dZ_eff collapse fails, and why one lever could never show it.

Panel (a) is the collapse as it looked with the friction knob alone: n against
dZ_eff, 2D and 3D at matched pressure, agreeing at 1.5 sigma. Respectable, and
it got BETTER when the overlap was sampled more densely -- normally a sign a
collapse is real.

Panel (b) adds the stiffness knob. Emod moves Z at fixed mu_g and fixed
pressure, so it reaches the same dZ_eff by an independent route. Those points
do not lie on the friction curve: gaps of +0.078, -0.055 and +0.042 (3.7, 2.9,
3.0 sigma), in BOTH directions, so no rescaling of Z_c can repair it.

Panel (c) is the reason, and the transferable lesson. Along the friction lever
n and dZ_eff are both driven by mu_g, so they move together whether or not one
causes the other -- a common-cause correlation is indistinguishable from a
collapse when you only have one knob. The stiffness lever separates them: it
moves dZ_eff over nearly the same range while n responds in the OPPOSITE
direction.

Usage:  python3 make_two_lever_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analyze_collapse_pooled as P
import make_collapse_matched as M

C_FR, C_ST, C_3D = 'tab:blue', 'tab:green', 'tab:red'


def friction_lever():
    mac, t1 = P.load('results_matched2d.csv'), P.load('tier1_matched2d_results.csv')
    out = []
    for mg in sorted({r['mu_g'] for r in mac}):
        tt = [r for r in t1 if r['mu_g'] == mg]
        if not tt:
            continue
        n, e = P.gate_and_fit([r for r in mac if r['mu_g'] == mg])
        if np.isfinite(n):
            out.append((float(np.mean([r['dZ_eff'] for r in tt])), n, e, mg))
    return sorted(out)


def stiffness_lever():
    import glob, os, re
    st = P.load('tier1_stiff_results.csv')
    for r in st:
        r['E'] = float(re.match(r'E([0-9.eE+-]+)_mu', r['label']).group(1))
    out = []
    for E in sorted({r['E'] for r in st}):
        for mg in sorted({r['mu_g'] for r in st}):
            dzr = [r for r in st if r['mu_g'] == mg and r['E'] == E]
            if not dzr:
                continue
            runs = []
            for p in glob.glob(os.path.join('sweep_stiffness', 'log.E*')):
                m = re.match(r'log\.E([0-9.eE+-]+)_mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$',
                             os.path.basename(p))
                if not m or float(m.group(1)) != E or float(m.group(2)) != mg:
                    continue
                q = P.parse_log(p)
                if q:
                    q['Tgran'] = float(m.group(3))
                    runs.append(q)
            if len({r['Tgran'] for r in runs}) < 3:
                continue
            n, e = P.gate_and_fit(runs)
            if np.isfinite(n):
                out.append((float(np.mean([r['dZ_eff'] for r in dzr])), n, e, mg, E))
    return sorted(out)


def main():
    FR, ST = friction_lever(), stiffness_lever()
    fx = np.array([a[0] for a in FR])
    fy = np.array([a[1] for a in FR])
    fe = np.array([a[2] for a in FR])

    mu3, n3, e3, dz3, _ = M.series('results_3d.csv', 'tier1_3d_results.csv')

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # (a) the one-lever picture
    a = ax[0]
    a.errorbar(fx, fy, yerr=fe, fmt='o', color=C_FR, capsize=3, label='2D (friction lever)')
    a.errorbar(dz3, n3, yerr=e3, fmt='s', color=C_3D, capsize=3, label='3D (friction lever)')
    a.set_xlabel(r'$\Delta Z_{\rm eff}$')
    a.set_ylabel('$n$')
    a.set_title('(a) one knob: looks like a collapse\n'
                r'2D vs 3D agree at 1.5$\sigma$', fontsize=9)
    a.legend(fontsize=7)

    # (b) add the orthogonal lever
    a = ax[1]
    a.plot(fx, fy, '-', color=C_FR, lw=1.2, zorder=1)
    a.errorbar(fx, fy, yerr=fe, fmt='o', color=C_FR, capsize=3,
               label='friction lever ($E$=1e5)', zorder=2)
    shown = False
    for d, n, e, mg, E in ST:
        if abs(E - 1.0e5) < 1:
            continue
        a.errorbar(d, n, yerr=e, fmt='D', color=C_ST, capsize=3, ms=6,
                   label=None if shown else 'stiffness lever', zorder=3)
        shown = True
        if fx.min() <= d <= fx.max():
            pred = float(np.interp(d, fx, fy))
            a.annotate('', xy=(d, n), xytext=(d, pred),
                       arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.1))
            a.annotate(f'{abs(n-pred)/e:.1f}$\\sigma$', (d, (n + pred) / 2),
                       fontsize=7, xytext=(5, -2), textcoords='offset points',
                       color='0.25')
    a.set_xlabel(r'$\Delta Z_{\rm eff}$')
    a.set_ylabel('$n$')
    a.set_title('(b) second knob: the collapse breaks\n'
                r'gaps both ways, RMS 3.2$\sigma$', fontsize=9)
    a.legend(fontsize=7)

    # (c) why one lever cannot decide
    a = ax[2]
    fr_mu = np.array([x[3] for x in FR])
    o = np.argsort(fr_mu)
    a.plot(fx[o], fy[o], 'o-', color=C_FR, label=r'friction: $\mu_g$ drives BOTH')
    for mg, mk in [(0.1, 'D-'), (0.3, 's-'), (1.0, '^-')]:
        pts = sorted([(d, n) for d, n, e, m, E in ST if m == mg])
        if len(pts) >= 2:
            a.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=C_ST,
                   ms=5, label=rf'stiffness at $\mu_g$={mg:g}')
    a.set_xlabel(r'$\Delta Z_{\rm eff}$')
    a.set_ylabel('$n$')
    a.set_title('(c) the two levers traverse the same axis\n'
                'with opposite slopes', fontsize=9)
    a.legend(fontsize=6.5)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig8_two_lever.{ext}', dpi=200, bbox_inches='tight')
    print('wrote fig8_two_lever.{pdf,png}')


if __name__ == '__main__':
    main()
