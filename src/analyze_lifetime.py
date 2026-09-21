"""Contact lifetimes, and the exponent alpha they imply.

WHY THIS EXISTS.  Sec. IV rules out the account in which n depends on the
single combination t_c*gdot = I*kappa^(-alpha), because the surface fit wants
alpha = 0.63 while a Hertzian contact gives 1/3 to 2/5.  That rests on an alpha
taken from contact mechanics rather than from the flow.  pair_style tracker
records, for every contact that breaks, the step it formed and how many steps it
lasted, so the mean contact duration is measurable here and alpha with it:

    alpha = -d ln <t_c> / d ln kappa      at fixed gdot and Theta.

WHAT THE RECORDS ARE AND ARE NOT.  Only contacts that break inside the
measurement window are written, so contacts still alive at the end are missing
and the mean is biased low.  The bias is the same at both stiffnesses at matched
strain, so it largely cancels in a ratio; the absolute durations should be read
as lower bounds.  Durations are in timesteps, and dt differs between moduli by
construction, so every duration is converted to time before anything is compared.

Files are large, so the reader stops after CAP records per run.  Records are
written in flushes covering the whole window, and taking whole flushes keeps the
sample unbiased in time.

Usage:  python3 analyze_lifetime.py      (run from the directory holding the sweeps)
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 'sweep_tracker'
RX = (r'dump\.track\.E(?P<E>[0-9.eE+-]+)_g(?P<g>[0-9.eE+-]+)'
      r'_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
CAP = 2_000_000
# A pair that overlaps for a single timestep has not made a contact in any sense
# this argument uses, since nothing accumulates across it.  Those grazings are a
# small tail everywhere except at the coldest target of the fast, soft series,
# where the bath is cooling rather than heating and they are most of the
# records.  Dropping them removes that one anomaly and moves alpha elsewhere by
# about 0.01.
MINSTEPS = 2
DT = {1.0e4: 2.5118864e-3, 1.0e5: 1.0e-3,
      1.0e6: 3.9810717e-4}            # in.granular_2d_stiff.py
P_CONF = 10.0


def durations(path, cap=CAP):
    """Contact durations in timesteps, reading whole flushes up to the cap."""
    out, buf, in_entries = [], [], False
    with open(path, errors='ignore') as fh:
        for line in fh:
            if line.startswith('ITEM:'):
                if in_entries and buf:
                    out.append(np.array(buf, dtype=np.float64))
                    buf = []
                    if sum(len(a) for a in out) >= cap:
                        break
                in_entries = line.startswith('ITEM: ENTRIES')
                continue
            if not in_entries:
                continue
            t = line.split()
            if len(t) == 4:
                buf.append(float(t[3]))
    if buf:
        out.append(np.array(buf, dtype=np.float64))
    return np.concatenate(out) if out else np.array([])


def main():
    runs = {}
    for p in sorted(glob.glob(f'{D}/dump.track.*')):
        m = re.match(RX, os.path.basename(p))
        if not m:
            continue
        key = (float(m.group('E')), float(m.group('g')), float(m.group('T')))
        d = durations(p)
        d = d[d >= MINSTEPS]
        if not d.size:
            continue
        runs.setdefault(key, []).append(d)

    if not runs:
        print('no tracker records found')
        return 1

    print('=' * 84)
    print('CONTACT LIFETIME   2D, mu_g = 0.3,  durations in time units')
    print('=' * 84)
    print(f'  {"E":>8}{"kappa":>9}{"gdot":>11}{"Theta set":>11}'
          f'{"records":>10}{"<t_c>":>10}{"median":>10}{"t_c*gdot":>11}')
    tc = {}
    for key in sorted(runs):
        E, g, T = key
        d = np.concatenate(runs[key]) * DT[E]        # steps -> time
        tc[key] = (float(d.mean()), float(np.median(d)), d.size)
        print(f'  {E:>8.0e}{E/P_CONF:>9.0f}{g:>11.3e}{T:>11g}'
              f'{d.size:>10d}{d.mean():>10.4f}{np.median(d):>10.4f}'
              f'{d.mean()*g:>11.2e}')

    print('\n  ALPHA, FROM HOW THE MEAN DURATION MOVES WITH STIFFNESS')
    print('  t_c ~ kappa^(-alpha), one decade in kappa per pair')
    es = sorted({k[0] for k in tc})
    if len(es) < 2:
        print('  need at least two moduli')
        return 1
    al = []
    for g in sorted({k[1] for k in tc}):
        for T in sorted({k[2] for k in tc}):
            have = [E for E in es if (E, g, T) in tc]
            if len(have) < 2:
                continue
            x = np.log(np.array(have))
            row = []
            for j in (0, 1):               # mean, then median
                y = np.log(np.array([tc[(E, g, T)][j] for E in have]))
                row.append(-np.polyfit(x, y, 1)[0])
            al.append(row)
            seq = ' -> '.join(f'{tc[(E, g, T)][0]:.4f}' for E in have)
            print(f'    gdot = {g:.3e}, Theta set = {T:<7g}  <t_c> {seq}'
                  f'   alpha = {row[0]:+.3f}   (median {row[1]:+.3f})')
    print(f'\n    fitted over kappa = {min(es)/P_CONF:.0f} to {max(es)/P_CONF:.0f}, '
          f'{len(es)} moduli')
    al = np.array(al)
    # The distribution is heavily skewed: rattling contacts dominate the count
    # and enduring ones the mean, so alpha is quoted from both.
    for j, lab in ((0, 'mean'), (1, 'median')):
        v = al[:, j]
        print(f'\n    alpha from the {lab}: {v.mean():.3f} +/- '
              f'{v.std(ddof=1)/np.sqrt(v.size):.3f} over {v.size} matched pairs, '
              f'spread {v.std(ddof=1):.3f}')
    print('\n  BETA, FROM HOW THE MEAN DURATION MOVES WITH TEMPERATURE')
    print('  t_c ~ Theta^(-beta).  A grain carried out of contact by its own')
    print('  agitation would take a time d/sqrt(Theta), so beta = 1/2 there.')
    import nfit as _nfit
    import glob as _glob
    th = {}
    for p in _glob.glob(f'{D}/log.*'):
        m = re.match(RX.replace('dump\\.track', 'log'), os.path.basename(p))
        if not m:
            continue
        q = _nfit.parse_log(p)
        if q:
            k = (float(m.group('E')), float(m.group('g')), float(m.group('T')))
            th.setdefault(k, []).append(q['Theta'])
    be = []
    for E in sorted({k[0] for k in tc}):
        for g in sorted({k[1] for k in tc if k[0] == E}):
            Ts = sorted({k[2] for k in tc if k[0] == E and k[1] == g})
            if len(Ts) < 3 or any((E, g, T) not in th for T in Ts):
                continue
            x = np.log([np.mean(th[(E, g, T)]) for T in Ts])
            y = np.log([tc[(E, g, T)][0] for T in Ts])
            b = -np.polyfit(x, y, 1)[0]
            be.append(b)
            print(f'    E = {E:.0e}, gdot = {g:.3e}   beta = {b:+.3f}'
                  f'   over Theta = {np.exp(x[0]):.2e} to {np.exp(x[-1]):.2e}')
    if be:
        be = np.array(be)
        print(f'\n    beta = {be.mean():.3f} +/- '
              f'{be.std(ddof=1)/np.sqrt(be.size):.3f}, spread {be.std(ddof=1):.3f}'
              f', over {be.size} series')
    print(f'\n    Hertzian impact at fixed speed: 2/5 = 0.400')
    print(f'    contact already loaded to P:    1/3 = 0.333')
    print(f'    what the surface fit of confound_joint.py wants: 0.63')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
