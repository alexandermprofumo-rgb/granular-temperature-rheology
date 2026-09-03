"""A fourth validity gate: is the packing still JAMMED?

nfit's three gates check the barostat, fixed-I, and thermostat control. None of
them notices the packing coming apart. Across the extended frictionless window
the coordination number falls monotonically with Theta -- 2D Z: 4.10 -> 3.00,
3D Z: 6.38 -> 3.83 -- so the warm setpoints are progressively UNJAMMING, and
"n rising with Theta" there is the approach to unjamming rather than a property
of dense-flow rheology. At the warm end 3D returns n = 0.160 +/- 0.020, which
is 0.3 sigma from 1/6, at Z = 4.4 in a system that needs Z = 6 to be rigid.

The pre-existing grids are exposed too: the production 2D grid runs to
Tgran = 0.03 (Z = 3.87 < 4) and the 3D grid to Tgran = 0.12 (Z = 5.20 < 6).
(Inferred -- the old decks did not write Z to the thermo, so this cannot be
checked directly on them. At strain 0.12 the packing has dilated less, so the
old Z is probably somewhat higher at the same setpoint.)

WHICH THRESHOLD? Isostaticity is friction-dependent: 2D counting gives
Z_iso = 4 frictionless (1 constraint per contact, 2 DOF per grain) and
Z_iso = 3 for non-sliding frictional contacts (2 constraints, 3 DOF); in 3D,
6 and 4. Rather than pick one, this reports the constraint list under three
thresholds and shows the movement.

Usage:  python3 analyze_jamming_gate.py
"""
import glob
import os
import re

import numpy as np

import nfit

RX = re.compile(r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def cells(d):
    out = {}
    for p in glob.glob(f'{d}/log.mu*'):
        m = RX.match(os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(float(m.group('mu')), []).append(q)
    return out


def jammed_setpoints(runs, zmin):
    """Setpoints surviving nfit's gates AND with mean Z >= zmin."""
    Ts, byT = nfit.surviving_setpoints(runs)
    keep = []
    for t in Ts:
        z = np.mean([x['Z'] for x in byT[t] if 'Z' in x]) if any('Z' in x for x in byT[t]) else np.nan
        if not np.isfinite(z) or z >= zmin:
            keep.append(t)
    return keep


def main():
    S = {'2D': (cells('sweep_steady2d'), 2), '3D': (cells('sweep_steady3d'), 3)}

    print('=' * 78)
    print('COORDINATION NUMBER ACROSS THE Theta WINDOW, EVERY CELL')
    print('=' * 78)
    for tag, (C, D) in S.items():
        print(f'\n{tag}   (Z_iso: {2*D} frictionless, {D+1} fully frictional)')
        Tall = sorted({round(r['Tgran'], 12) for m in CORE if m in C for r in C[m]})
        print('  mu_g \\ Tgran ' + ''.join(f'{t:>8g}' for t in Tall))
        for mg in CORE:
            if mg not in C:
                continue
            byT = {}
            for r in C[mg]:
                byT.setdefault(round(r['Tgran'], 12), []).append(r)
            row = ''
            for t in Tall:
                if t not in byT:
                    row += f'{"-":>8}'
                else:
                    z = np.mean([x.get('Z', np.nan) for x in byT[t]])
                    row += f'{z:>8.2f}' if np.isfinite(z) else f'{"-":>8}'
            print(f'  {mg:<12g}' + row)

    print()
    print('=' * 78)
    print('THE CONSTRAINT LIST UNDER THREE JAMMING THRESHOLDS')
    print('=' * 78)
    for tag, (C, D) in S.items():
        pred = 0.25 if D == 2 else 1.0 / 6.0
        print(f'\n--- {tag} ---')
        print(f'{"threshold":<22}{"k(mu=0)":>9}{"n(mu=0)":>20}{"vs 1/(2D)":>12}'
              f'{"peak":>10}{"at":>7}{"rise":>9}')
        for name, zmin in (('none (as published)', -1.0),
                           (f'Z >= {D+1} (frictional)', float(D + 1)),
                           (f'Z >= {2*D} (frictionless)', float(2 * D))):
            keep = {m: jammed_setpoints(C[m], zmin) for m in CORE if m in C}
            pool = [C[m] for m in keep if len(keep[m]) >= 4]
            if not pool:
                print(f'{name:<22}   nothing survives')
                continue
            los, his = [], []
            for m in keep:
                if len(keep[m]) < 4:
                    continue
                Ts, byT = nfit.surviving_setpoints(C[m], restrict_to=keep[m])
                th = [x['Theta'] for t in Ts for x in byT[t] if x['Theta'] > 0]
                if th:
                    los.append(min(th)); his.append(max(th))
            if not los or max(los) >= min(his):
                print(f'{name:<22}   no common Theta range')
                continue
            T0 = float(np.sqrt(max(los) * min(his)))
            R = {}
            for m in keep:
                if len(keep[m]) < 4:
                    continue
                r = nfit.fit_local_sys(C[m], T0, restrict_to=keep[m])
                if r:
                    R[m] = r
            if 0.0 not in R or len(R) < 4:
                print(f'{name:<22}   too few cells')
                continue
            b = R[0.0]
            pk = max(R, key=lambda m: R[m]['n'])
            e = float(np.hypot(R[pk]['tot'], b['tot']))
            print(f'{name:<22}{b["k"]:>9}{b["n"]:>+13.4f}+/-{b["tot"]:<6.4f}'
                  f'{abs(b["n"]-pred)/b["tot"]:>10.1f}s{R[pk]["n"]:>10.4f}'
                  f'{pk:>7g}{(R[pk]["n"]-b["n"])/e:>8.1f}s')
        print(f'   (Theta_0 is re-derived for each threshold, so n moves partly '
              f'because the window moves)')


MARGIN = 0.02   # 'marginal' = within this of the threshold
RX_ANY = re.compile(r'log\..*_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')


def _any_cells(d):
    """Group a sweep's runs by every label except the setpoint and the seed.

    cells() above only matches log.mu*, which is the friction-scan naming. The
    lever sweeps prefix the stiffness, pressure or tangential stiffness, so they
    would silently contribute nothing to a survey that used it.
    """
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        b = os.path.basename(p)
        m = RX_ANY.match(b)
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        out.setdefault(b[:m.start('T')], []).append(q)
    return out


def majority_rule_audit(sweeps=None, z=None):
    """What the majority rule catches that per-run filtering alone does not.

    Sec. II.D of the manuscript states that filtering runs without the majority
    rule removes 5 sub-isostatic runs and wrongly admits 10 marginal setpoints.
    That was a docstring in nfit.py and nothing computed it; this does.

    A setpoint is "wrongly admitted" when a minority of its runs are jammed, so
    run-filtering alone would keep it on the strength of that minority while the
    setpoint mean correctly rejects it.
    """
    if sweeps is None:
        sweeps = [('sweep_steady2d', 3.0), ('sweep_steady3d', 4.0),
                  ('sweep_lev_E', 3.0), ('sweep_lev_P', 3.0),
                  ('sweep_lev_kt', 3.0), ('sweep_size2d', 3.0),
                  ('sweep_restit', 3.0), ('sweep_iscan2', 3.0),
                  ('sweep_lev_E3d', 4.0), ('sweep_iscan3d', 4.0)]
    removed = admitted = total = 0
    for name, zmin in sweeps:
        zz = z if z is not None else zmin
        for runs in _any_cells(name).values():
            # nfit's other three gates first: the survey is over runs that
            # would otherwise be used, not over everything on disk.
            Ts, byT = nfit.surviving_setpoints(runs)
            for t in Ts:
                rs = [r for r in byT[t] if r.get('Z') is not None]
                if not rs:
                    continue
                total += len(rs)
                jam = [r for r in rs if r['Z'] >= zz]
                if len(jam) * 2 > len(rs):
                    removed += len(rs) - len(jam)      # majority holds: prune
                elif jam and any(abs(r['Z'] - zz) <= MARGIN for r in jam):
                    admitted += 1                       # marginal minority
    print()
    print('=' * 78)
    print('MAJORITY RULE: what run-filtering alone would do')
    print('=' * 78)
    print(f'  surveyed {total} runs across {len(sweeps)} gated sweeps')
    print(f'  sub-isostatic runs removed where the setpoint keeps a majority: {removed}')
    print(f'  setpoints run-filtering would wrongly admit on a minority:      {admitted}')
    print('  With the majority rule all of the second group are rejected and')
    print('  all of the first are removed.')


if __name__ == '__main__':
    main()
    majority_rule_audit()
