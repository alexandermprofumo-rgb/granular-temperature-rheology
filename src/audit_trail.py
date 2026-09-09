"""An audit trail for every core cell, and a check on the error model itself.

WHY THIS EXISTS.  Two independent reviewers asked for the same two things.

FIRST, AN AUDIT TRAIL.  The pipeline is layered -- validity gating, adaptive
degree selection, a local fit at a stated Theta_0, a calibrated jackknife -- and
a reader cannot currently see what any single cell contributed.  For each cell
this prints the runs parsed, the setpoints surviving the gate, the Theta window
actually fitted, the polynomial degree the adaptive fit chose, and n with its
statistical and systematic parts separated rather than combined.

SECOND, THE RESAMPLING BLOCK.  The quoted errors come from a jackknife over
SETPOINTS and a Monte Carlo that draws parametric replicas from the fitted
polynomial.  Neither can see variation between independent packings, because
neither resamples packings: replicas drawn from a fitted curve reproduce that
curve's seed by construction.  With two or three packings per cell a bootstrap
over packings has too few blocks to be worth running, but the same information
is available directly.  Fitting each packing separately gives a difference whose
expected size is known: if the quoted error sigma is honest, then

    z = (n_1 - n_2) / (sqrt(2) * sigma)

is standard normal across cells.  A width above one means the errors are
understated by the factor reported, and the headline significances should be
divided by it.  This is the cheapest honest test of the point at issue, and it
uses no data the project does not already have.

Usage:  python3 audit_trail.py
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfit

RXM = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'
SWEEPS = (('2D', 'sweep_steady2d', 3.0), ('3D', 'sweep_steady3d', 4.0))
CANON = 8.4664e-04


def cells(d):
    out = {}
    for p in glob.glob(f'{d}/log.*'):
        m = re.match(RXM, os.path.basename(p))
        if not m:
            continue
        q = nfit.parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        q['seed'] = int(m.group('s'))
        out.setdefault(float(m.group('mu')), []).append(q)
    return out


def gated(runs, zmin):
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    return [t for t in Ts
            if not [x['Z'] for x in byT[t] if 'Z' in x]
            or np.mean([x['Z'] for x in byT[t] if 'Z' in x]) >= zmin]


def window(runs, keep, zmin):
    _, byT = nfit.surviving_setpoints(runs, z_min=zmin, restrict_to=keep)
    th = [r['Theta'] for t in keep for r in byT[t] if r['Theta'] > 0]
    return (min(th), max(th)) if th else None


def fit(runs, keep, zmin):
    return nfit.fit_local_sys(runs, CANON, fn=nfit.fit_local_adaptive,
                              restrict_to=keep, z_min=zmin)


KK_P = {'2D': 1.0 / 8.0, '3D': 1.0 / 6.0}   # Kim & Kamrin's fitted exponents


def main():
    zs = []
    domin = {}
    frictionless = {}
    for tag, d, zmin in SWEEPS:
        C = cells(d)
        if not C:
            print(f'{d} not unpacked; skipping {tag}.')
            continue
        print('=' * 100)
        print(f'{tag}:  AUDIT TRAIL, CORE FRICTION SCAN')
        print('=' * 100)
        print(f'  {"mu_g":>6}{"runs":>6}{"setpts":>8}{"gated":>7}{"deg":>5}'
              f'{"Theta window":>26}{"n":>9}{"stat":>9}{"sys":>9}{"total":>9}')
        for m in sorted(C):
            runs = C[m]
            allT = sorted({r['Tgran'] for r in runs})
            keep = gated(runs, zmin)
            if len(keep) < 4:
                print(f'  {m:>6g}{len(runs):>6}{len(allT):>8}{len(keep):>7}'
                      f'   not fittable')
                continue
            w = window(runs, keep, zmin)
            r = fit(runs, keep, zmin)
            if not r:
                continue
            deg = r.get('deg', r.get('degree', '-'))
            stat, sysd = r.get('tot', np.nan), r.get('sys_deg', 0.0)
            tot = float(np.hypot(stat, sysd))
            print(f'  {m:>6g}{len(runs):>6}{len(allT):>8}{len(keep):>7}'
                  f'{str(deg):>5}{f"[{w[0]:.2e}, {w[1]:.2e}]":>26}'
                  f'{r["n"]:>9.4f}{stat:>9.4f}{sysd:>9.4f}{tot:>9.4f}')
            domin.setdefault(tag, []).append((m, stat, sysd))
            if m == 0.0:
                frictionless[tag] = (r['n'], tot)

            # per-packing refit: the block the error model never resamples
            seeds = sorted({x['seed'] for x in runs})
            per = {}
            for s in seeds:
                sub = [x for x in runs if x['seed'] == s]
                k = [t for t in keep if any(x['Tgran'] == t for x in sub)]
                if len(k) < 4:
                    continue
                q = fit(sub, k, zmin)
                if q:
                    per[s] = q['n']
            if len(per) >= 2:
                v = list(per.values())
                for i in range(len(v)):
                    for j in range(i + 1, len(v)):
                        if tot > 0:
                            zs.append((v[i] - v[j]) / (np.sqrt(2) * tot))

    print('\n' + '=' * 100)
    print('WHICH ERROR COMPONENT DOMINATES, AND WHERE')
    print('=' * 100)
    print('  The paper quotes the two combined, so a reader cannot otherwise')
    print('  see that the dimensions are limited by different things.\n')
    for tag, rows in domin.items():
        nstat = sum(1 for _, st, sy in rows if st > sy)
        worst = max(rows, key=lambda r: (r[2] / r[1]) if r[1] > 0 else 0)
        print(f'  {tag}: statistical dominates in {nstat} of {len(rows)} cells, '
              f'systematic in {len(rows) - nstat}.')
        print(f'      largest systematic-to-statistical ratio '
              f'{worst[2]/worst[1]:.1f} at mu_g = {worst[0]:g} '
              f'({worst[2]:.4f} against {worst[1]:.4f}).')

    print('\n' + '=' * 100)
    print('THE FRICTIONLESS VALUES AGAINST THE PUBLISHED EXPONENTS')
    print('=' * 100)
    for tag, (n0, e0) in frictionless.items():
        print(f'  {tag}: n(mu_g=0) = {n0:.4f} +/- {e0:.4f}, below the fitted '
              f'p = {KK_P[tag]:.4f} by a factor {KK_P[tag]/n0:.1f}')
    print('  These are the factors quoted in the abstract and conclusion; they')
    print('  are ratios to the FITTED exponents, not to the geometric estimate.')

    print('\n' + '=' * 100)
    print('IS THE QUOTED ERROR CONSISTENT WITH THE SPREAD BETWEEN PACKINGS?')
    print('=' * 100)
    if len(zs) < 4:
        print('  too few cells with two fittable packings.')
        return 0
    a = np.array(zs)
    print(f'  {len(a)} packing pairs across both dimensions.')
    print(f'  z = (n_i - n_j)/(sqrt(2) sigma):  mean {a.mean():+.2f}, '
          f'RMS {np.sqrt((a**2).mean()):.2f}, max |z| {np.abs(a).max():.2f}')
    r = float(np.sqrt((a ** 2).mean()))
    if r <= 1.3:
        print('  The quoted errors cover the spread between packings.  No')
        print('  inflation is indicated.')
    else:
        print(f'  The spread exceeds the quoted errors by a factor {r:.2f}.')
        print(f'  Significances built on these errors should be divided by')
        print(f'  {r:.2f} until more packings are run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
