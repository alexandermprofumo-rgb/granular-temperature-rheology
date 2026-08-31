"""The 3D three-channel table under the CURRENT frame policy.

WHY. Earlier work compared the 2D channel decomposition (extracted
under the corrected "whole measurement half" policy) against 3D numbers still
produced by analyze_channels_3d_steady.py with keep_last=8 -- a fact that
script's own header records. "In 3D the same inversion occurs but is weaker
(C_t reaches 0.40 at mu_g = 1, not 0.81)" is therefore a cross-policy
comparison, and the 2026-08-25 audit flagged it. chan_steady3d_half.csv already
exists under the current policy (analyze_lev_E3d builds it), so the table can
simply be regenerated.

In 3D the exact identity is  mu = D*A_c + D*A_n + mu_t  (D = 3), so
    C_c = -(D/mu) dA_c/dlnTheta,  C_n = -(D/mu) dA_n/dlnTheta,
    C_t = -(1/mu) d(mu_t)/dlnTheta,   and  n = C_c + C_n + C_t.

Usage:  python3 analyze_channels_3d_half.py
"""
import csv
import json
import numpy as np

import nfit
import constraints_final as CF

D, NGRAIN, ZMIN = 3, 4000, 4.0
T0 = 8.466e-4


def loc(x, y, x0, deg=2):
    A = np.vander(np.asarray(x, float) - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(c[0]), float(c[1])


def main():
    rows = []
    for r in csv.DictReader(open('chan_steady3d_half.csv')):
        rows.append({k: (v if k == 'seed' else float(v)) for k, v in r.items()})
    C = CF.cells('sweep_steady3d')
    keep = {m: {round(t, 12) for t in CF.jammed(C[m], ZMIN)} for m in C}

    grp = {}
    for r in rows:
        m = r['mu_g']
        if m in keep and round(r['Tgran'], 12) in keep[m] and r['Theta'] > 0:
            grp.setdefault(m, []).append(r)

    print('3D three-channel decomposition, whole-measurement-half policy, '
          f'jamming-gated Z >= {ZMIN}, Theta_0 = {T0:.3e}')
    print(f'  {"mu_g":>6}{"C_c":>10}{"C_n":>10}{"C_t":>10}{"n = sum":>10}'
          f'{"  shares c / n / t":>22}')
    out = []
    for m in sorted(grp):
        rs = grp[m]
        if len(rs) < 8:
            continue
        x = np.log([r['Theta'] for r in rs]); x0 = np.log(T0)
        mu0 = loc(x, [r['mu'] for r in rs], x0)[0]
        if mu0 <= 0:
            continue
        Cc = -D * loc(x, [r['A_c'] for r in rs], x0)[1] / mu0
        Cn = -D * loc(x, [r['A_n'] for r in rs], x0)[1] / mu0
        Ct = -loc(x, [r['mu_t'] for r in rs], x0)[1] / mu0
        tot = Cc + Cn + Ct
        sh = (f'{Cc/tot:.2f} / {Cn/tot:.2f} / {Ct/tot:.2f}'
              if abs(tot) > 1e-9 else '--')
        print(f'  {m:>6g}{Cc:>10.4f}{Cn:>10.4f}{Ct:>10.4f}{tot:>10.4f}{sh:>22}')
        out.append([m, Cc, Cn, Ct])
    json.dump(out, open('channels3d_half.json', 'w'), indent=1)
    print('\n  cached -> channels3d_half.json')
    print('  Under keep_last=8 the 3D grip share reaches 0.40 at mu_g = 1,')
    print('  against 0.81 in 2D on the whole-half policy.')
    print('  Compare the 3D share above, which is now on the same policy as 2D.')


if __name__ == '__main__':
    main()
