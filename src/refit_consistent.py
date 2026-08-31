"""
Refit n(mu_g) for every sweep under ONE principled, per-cell setpoint criterion.

WHY THIS EXISTS
n is obtained by fitting ln(mu) against ln(Theta) at FIXED inertial number I,
scanning Theta via the thermostat setpoint. Two things can silently break that:

 1. I stops being fixed. At the coldest setpoints the system creeps rather than
    flows and I runs high -- +35% in the 2D Pconf=2 sweep, +9% in 2D Pconf=10,
    +6% on average in 3D. Those points violate the premise of the measurement.

 2. The packing leaves the dense regime. At the hottest setpoints it dilates:
    in 2D Pconf=2 the contact count falls 33% from its plateau, heading toward
    collisional flow, which is a different rheology.

The published 2D analysis handled both by hard-coding CORE2D = {5e-4, 2e-3,
6e-3, 1.5e-2}, and that window is well chosen -- it is exactly the plateau.
But the 3D analysis passed core=None (see make_collapse_figure.py), so 3D was
fitted over ALL setpoints including ones 2D would have rejected. Every
cross-dimension claim therefore compared unlike windows.

Rather than hard-code a different window per sweep, this applies the same two
physical tests cell by cell:

    keep a setpoint iff  |I/median(I) - 1| <= 3%   and   n_contacts >= 80% of
    the cell's maximum

and refits. Cells left with fewer than 3 setpoints are reported as unusable
rather than fitted.

ROBUSTNESS
Varying NC_FLOOR over 0.0-0.90 leaves both Pconf=10 sweeps unchanged:
2D n(0)=0.061 peak 0.174 at mu_g=0.15; 3D n(0)=0.062 peak 0.252 at mu_g=0.2.
The Pconf=2 production sweep is NOT stable under the same variation -- its
frictionless exponent moves 0.100 -> 0.048 -> undefined and its peak jumps
between mu_g=2 and mu_g=0.3. That instability has a physical cause: at Pconf=2
the 2D packing is hypostatic (Z=3.30 vs isostatic 4) and dilates across the
Theta scan, so the fitting window changes which state is being measured. At
Pconf=10 both dimensions are isostatic and the packing is stable throughout.

Usage:  python3 refit_consistent.py
"""
import csv
import numpy as np

I_TOL = 0.03      # fractional drift in I that still counts as "fixed I"
NC_FLOOR = 0.80   # contact count relative to the cell max: below this the
                  # packing has dilated out of the dense plateau. The Pconf=10
                  # results are INSENSITIVE to this value anywhere in 0.0-0.90;
                  # only the hypostatic Pconf=2 production sweep depends on it,
                  # which is itself the finding (see docstring).
MIN_PTS = 3

SWEEPS = [
    ('results.csv', '2D  Pconf=2  (production)'),
    ('results_matched2d.csv', '2D  Pconf=10 (matched)'),
    ('results_3d.csv', '3D  Pconf=10'),
]


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
    m = np.isfinite(th) & np.isfinite(mu) & (th > 0) & (mu > 0)
    if m.sum() < MIN_PTS:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])


def cell(rows):
    """Group one (sweep, mu_g) cell by setpoint and apply both physical cuts."""
    byT = {}
    for r in rows:
        byT.setdefault(r['Tgran'], []).append(r)
    Ts = sorted(byT)
    I = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
    NC = {t: np.mean([r['n_contacts'] for r in byT[t]]) for t in Ts}
    Imed = np.median(list(I.values()))
    ncmax = max(NC.values())
    keep, drop = [], []
    for t in Ts:
        bad = []
        if abs(I[t] / Imed - 1) > I_TOL:
            bad.append(f'I{(I[t]/Imed-1)*100:+.0f}%')
        if NC[t] < NC_FLOOR * ncmax:
            bad.append(f'nc{NC[t]/ncmax*100:.0f}%')
        (drop if bad else keep).append((t, ','.join(bad)))
    return keep, drop, byT


def main():
    table = {}
    for f, lab in SWEEPS:
        rs = load(f)
        table[lab] = {}
        print(f'=== {lab} ===')
        for mg in sorted(set(r['mu_g'] for r in rs)):
            rows = [r for r in rs if r['mu_g'] == mg]
            keep, drop, byT = cell(rows)
            sel = [r for t, _ in keep for r in byT[t]]
            n, e = fit([r['Theta'] for r in sel], [r['mu'] for r in sel])
            table[lab][mg] = (n, e, len(keep))
            d = ' '.join(f'{t:.0e}[{why}]' for t, why in drop) or '-'
            if len(keep) < MIN_PTS:
                print(f'  mu_g={mg:<6g} UNUSABLE ({len(keep)} setpoints left)  dropped: {d}')
            else:
                print(f'  mu_g={mg:<6g} n={n:+.3f}+/-{e:.3f}  [{len(keep)} pts]  dropped: {d}')
        print()

    print('=== side by side (consistent criterion) ===')
    labs = [l for _, l in SWEEPS]
    allmu = sorted({m for l in labs for m in table[l]})
    print(f'{"mu_g":>7}' + ''.join(f'{l[:12]:>18}' for l in labs))
    for mg in allmu:
        line = f'{mg:>7g}'
        for l in labs:
            if mg in table[l] and np.isfinite(table[l][mg][0]):
                n, e, _ = table[l][mg]
                line += f'{n:>11.3f}+/-{e:.3f}'
            else:
                line += f'{"--":>18}'
        print(line)

    print()
    for l in labs:
        v = {m: t for m, t in table[l].items() if np.isfinite(t[0])}
        if not v:
            continue
        pk = max(v, key=lambda k: v[k][0])
        z = v.get(0.0, (np.nan, np.nan, 0))
        print(f'{l:28s} n(mu_g=0)={z[0]:.3f}   peak {v[pk][0]:.3f}+/-{v[pk][1]:.3f} at mu_g={pk:g}')
    print('\ngeometric theory: n = 1/(2D)  ->  2D 0.250, 3D 0.167')


if __name__ == '__main__':
    main()
