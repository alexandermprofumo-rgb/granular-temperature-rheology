"""
Replica audit: do independent sweeps of the SAME physical state agree?

Several sweeps in this project overlap in configuration. They were run at
different times, from different data files and seeds, through different input
decks, and analysed by different scripts. Physically they are the same state,
so they must give the same n within the error budget. Any pair that does not
localises a defect -- in a deck, a gate, or an analysis path -- and does it
without needing new simulations.

The replicas, and what each isolates:

  matched2d  vs piscan(P=10)
      identical physics (gdot=1e-3, tdamp=0.5, Pconf=10, E=1e5, N=4000)
      reached through two different input decks -- in.granular_2d and
      in.granular_2d_piscan. Isolates deck differences and the Theta-window
      systematic, since piscan later gained 3 extra setpoints and 2 extra
      seeds at mu_g = 0.1 and 1.0.

  matched2d  vs stiffness(E=1e5)
      the Emod sweep's reference modulus IS the production modulus, so its
      E=1e5 column should reproduce matched2d. Isolates the scaled-timestep
      machinery in in.granular_2d_stiff (dt and step counts are rescaled as
      E^(-2/5); at E=1e5 the scaling is the identity, so any disagreement is
      a bug in the rescaling rather than physics).

  matched2d  vs steadystate
      NOT a replica -- 12.5x longer strain. Included as a control: it should
      differ only by the transient, already measured at ~0 (mu_g=0.1) and
      1.3 sigma (mu_g=0.3). A larger difference here would mean the
      "transient does not bias n" result is wrong.

Every comparison uses nfit with ONE policy, and for the deck comparisons also
restricts both sides to their COMMON setpoints -- otherwise the comparison
silently inherits the window systematic as a bias.

Usage:  python3 replica_audit.py
"""
import numpy as np

import nfit


def cells_by_mu(pattern, rx, tgran_from=None):
    raw = nfit.load_sweep(pattern, rx, tgran_from)
    out = {}
    for key, runs in raw.items():
        d = dict(key)
        out.setdefault(float(d['mu']), []).extend(runs)
    return out


def compare(name_a, A, name_b, B, common_window=True):
    print(f'\n=== {name_a}  vs  {name_b} ===')
    mus = sorted(set(A) & set(B))
    if not mus:
        print('   no shared friction values')
        return []
    hdr = (f'{"mu_g":>6}{name_a[:14]:>26}{name_b[:14]:>26}'
           f'{"diff":>10}{"n_sig":>8}')
    print(hdr)
    print('-' * len(hdr))
    devs = []
    for mg in mus:
        ra_all, rb_all = A[mg], B[mg]
        win = None
        if common_window:
            ta = {round(r['Tgran'], 12) for r in ra_all}
            tb = {round(r['Tgran'], 12) for r in rb_all}
            win = ta & tb
            if len(win) < 3:
                print(f'{mg:>6g}   <3 shared setpoints, skipped')
                continue
        ra = nfit.fit_n(ra_all, restrict_to=win)
        rb = nfit.fit_n(rb_all, restrict_to=win)
        if ra is None or rb is None:
            print(f'{mg:>6g}{"VOID" if ra is None else nfit.fmt(ra):>26}'
                  f'{"VOID" if rb is None else nfit.fmt(rb):>26}')
            continue
        d = rb['n'] - ra['n']
        s = float(np.hypot(ra['tot'], rb['tot']))
        z = abs(d) / s if s > 0 else np.nan
        devs.append(z)
        flag = '' if z < 2 else '   <-- DISAGREE'
        print(f'{mg:>6g}'
              f'{ra["n"]:>17.4f}+/-{ra["tot"]:.4f}'
              f'{rb["n"]:>17.4f}+/-{rb["tot"]:.4f}'
              f'{d:>+10.4f}{z:>8.1f}{flag}')
    if devs:
        print(f'   RMS discrepancy {np.sqrt(np.mean(np.array(devs)**2)):.2f} '
              f'sigma over {len(devs)} frictions '
              f'(errors include the setpoint-jackknife systematic)')
    return devs


def main():
    m2 = cells_by_mu('sweep_matched2d/log.mu*',
                     r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    pi10 = cells_by_mu(
        'sweep_piscan/log.P10.0_mu*',
        r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
        tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    stif = cells_by_mu(
        'sweep_stiffness/log.E1.0e5_mu*',
        r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
    stdy = cells_by_mu('sweep_steadystate/log.mu*',
                       r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')

    print('REPLICA AUDIT -- same physical state, independent sweeps')
    print('one gating policy (nfit); deck comparisons use COMMON setpoints only')
    all_dev = []
    all_dev += compare('matched2d', m2, 'piscan P=10', pi10)
    all_dev += compare('matched2d', m2, 'stiffness E=1e5', stif)
    print('\n--- control (NOT a replica: 12.5x strain) ---')
    compare('matched2d', m2, 'steadystate', stdy)

    if all_dev:
        a = np.array(all_dev)
        print(f'\nOVERALL across true replicas: RMS {np.sqrt(np.mean(a**2)):.2f} '
              f'sigma, max {a.max():.1f} sigma, {int((a >= 2).sum())}/{len(a)} '
              f'above 2 sigma')
        print('   RMS <~1 means the error budget is honest and the sweeps are '
              'consistent.')
        print('   RMS >~2 means a defect remains -- and it is in the analysis '
              'or a deck, not in the physics.')

    # the window systematic, quantified across the whole project
    print('\n--- size of the setpoint-window systematic, all cells ---')
    tot = []
    for name, C in (('matched2d', m2), ('piscan P=10', pi10),
                    ('stiffness E=1e5', stif)):
        for mg, runs in sorted(C.items()):
            r = nfit.fit_n(runs)
            if r and np.isfinite(r['sys']):
                tot.append((r['sys'], r['stat'], name, mg))
    if tot:
        rat = np.array([a / b for a, b, _, _ in tot if b > 0])
        print(f'   sigma_sys / sigma_stat : median {np.median(rat):.2f}, '
              f'range {rat.min():.2f}-{rat.max():.2f}, over {len(rat)} cells')
        print('   (a ratio near or above 1 means quoting statistical errors '
              'alone understates the uncertainty)')


if __name__ == '__main__':
    main()
