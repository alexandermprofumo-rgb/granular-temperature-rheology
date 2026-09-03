"""The constraint list, final: steady state + jamming gate + calibrated errors.

Every number here comes from one policy so the results are internally
consistent:
  * strain 1.0 (0.5 equilibration + 0.5 measurement), not 0.12
  * nfit's four gates: barostat, fixed-I, thermostat, and Z >= D+1 (jamming)
  * the setpoint jackknife calibrated against its own null expectation
  * one Theta_0 per comparison, common to every cell in it

Usage:  python3 constraints_final.py
"""
import glob
import os
import re
import sys

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


def jammed(runs, zmin):
    """Setpoints that survive the gates, with the jamming test applied twice.

    PER RUN (z_min, added 2026-08-28): an individual run below the isostatic
    count is discarded before any setpoint statistic is formed. Z >= D+1 is an
    absolute physical threshold, not a relative one, so it is meaningful per
    run -- see nfit.surviving_setpoints. Four 2D runs out of 531 fail this
    while their setpoint mean passes, all at the coldest Tgran, and each
    carried mu 19-27% above its partners at the highest-leverage end of the fit.

    PER SETPOINT (the mean test below): a setpoint where the pack has genuinely
    unjammed is dropped entirely. Both tests are needed -- the first removes a
    bad realisation, the second removes a bad state.
    """
    Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
    keep = []
    for t in Ts:
        z = [x['Z'] for x in byT[t] if 'Z' in x]
        if not z or np.mean(z) >= zmin:
            keep.append(t)
    return keep


def sig(d, e):
    return abs(d) / e


def main():
    S = {'2D': (cells('sweep_steady2d'), 2, 0.25),
         '3D': (cells('sweep_steady3d'), 3, 1.0 / 6.0)}
    K = {tag: {m: jammed(C[m], D + 1) for m in CORE if m in C}
         for tag, (C, D, _) in S.items()}

    # one Theta_0 common to every cell in BOTH dimensions
    pool, los, his = [], [], []
    for tag, (C, D, _) in S.items():
        for m in CORE:
            ts = K[tag].get(m, [])
            if len(ts) < 4:
                continue
            _, byT = nfit.surviving_setpoints(C[m], restrict_to=ts)
            th = [x['Theta'] for t in ts for x in byT.get(t, []) if x['Theta'] > 0]
            if th:
                los.append(min(th)); his.append(max(th))
    T0 = float(np.sqrt(max(los) * min(his)))
    print(f'common Theta_0 = {T0:.3e}   (jamming gate Z >= D+1)\n')

    # ANALYSIS_PROTOCOL Amendment 6 (2026-08-26): the ADAPTIVE-degree fit is
    # PRIMARY; the quadratic is the cross-check. Both are always printed.
    #
    # Why adaptive leads: a quadratic's slope at Theta_0 depends on the RANGE
    # fitted when mu(Theta) is curved, and it is curved at up to 12.7 sigma per
    # cell. Adaptive degree removes the misspecification rather than working
    # around it, and -- decisively -- it withdraws the dimension effect on OWN
    # windows (chi2/dof 7.61 / 4.80) with no window matching at all, so the
    # conclusion does not rest on the more contestable operation.
    #
    # THE COST, stated because it must be: fit_local_sys cannot calibrate the
    # setpoint jackknife for a variable-degree design (the analytic null
    # assumes the fixed quadratic), so the adaptive column carries the RAW
    # jackknife, inflated ~1.1-1.8x relative to the calibrated excess. Its
    # errors are conservative for two reasons at once -- a genuine one (more
    # parameters) and a technical one (no calibration). An adaptive error and a
    # quadratic error are NOT the same estimator; do not difference them.
    FITS = (('adaptive', nfit.fit_local_adaptive), ('quadratic', nfit.fit_local))
    RR = {}
    for name, fn in FITS:
        RR[name] = {}
        for tag, (C, D, pred) in S.items():
            RR[name][tag] = {}
            for m in CORE:
                ts = K[tag].get(m, [])
                if len(ts) < 4:
                    continue
                r = nfit.fit_local_sys(C[m], T0, fn=fn, restrict_to=ts,
                                       z_min=D + 1)
                if r:
                    RR[name][tag][m] = r
    R = RR['adaptive']        # PRIMARY: every constraint below uses this

    for tag, (C, D, pred) in S.items():
        print(f'=== {tag} ===   [PRIMARY adaptive | cross-check quadratic]')
        print(f'{"mu_g":>6}{"k":>4}{"deg":>5}{"n +/- (stat+win) +/- deg":>34}'
              f'{"vs 1/(2D)":>10}{"n (quadratic)":>22}')
        for m in CORE:
            ra = RR['adaptive'][tag].get(m)
            rq = RR['quadratic'][tag].get(m)
            if not ra:
                print(f'{m:>6g}{len(K[tag].get(m, [])):>4}   too few jammed setpoints')
                continue
            deg = ra.get('deg', 2)
            # THE COMBINED ERROR. tot is stat + window systematic. sys_deg is
            # the model-selection systematic, held in its own column because
            # the audit asked for it separately -- but a SIGNIFICANCE must use
            # both, or the number improves merely by relocating a systematic.
            # 3D mu_g = 0 is the worked example: tot alone reads 35.8 sigma,
            # tot with sys_deg reads 18.4.
            ra['comb'] = float(np.hypot(ra['tot'], ra.get('sys_deg', 0.0)))
            qs = f'{rq["n"]:+.4f}+/-{rq["tot"]:.4f}' if rq else ''
            print(f'{m:>6g}{ra["k"]:>4}{deg:>5}{ra["n"]:>+11.4f}+/-{ra["tot"]:<6.4f}'
                  f'+/-{ra.get("sys_deg", 0.0):<6.4f}'
                  f'{sig(ra["n"] - pred, ra["comb"]):>8.1f}s{qs:>22}')
        print()
    ndeg = sum(1 for tag in RR['adaptive'] for m in RR['adaptive'][tag]
               if RR['adaptive'][tag][m].get('deg', 2) > 2)
    ntot = sum(len(RR['adaptive'][tag]) for tag in RR['adaptive'])
    print(f'  degree > 2 selected in {ndeg} of {ntot} cells '
          f'(sequential F-test, alpha = 0.05, max degree 4)')
    print('  the jackknife is run at FROZEN degree, so the analytic null applies '
          'at that\n  degree and the window systematic is calibrated, not raw. '
          'Model selection is\n  quoted separately as +/- deg; every significance '
          'above uses both in quadrature.\n')

    print('=' * 70)
    a, b = R['2D'][0.0], R['3D'][0.0]
    d = b['n'] - a['n']
    e = float(np.hypot(a.get('comb', a['tot']), b.get('comb', b['tot'])))
    print(f'C1  frictionless baselines equal: {d:+.4f} +/- {e:.4f} -> {sig(d,e):.1f} sigma')
    print(f'    geometric predicts {1/6-0.25:+.4f} -> excluded at '
          f'{sig(d-(1/6-0.25), e):.1f} sigma BY THE DIFFERENCE ALONE')
    print(f'C2  2D {a["n"]:.4f}+/-{a["tot"]:.4f}+/-{a.get("sys_deg",0):.4f} vs 0.25   -> '
          f'{sig(a["n"]-0.25,a.get("comb", a["tot"])):.1f} sigma')
    print(f'    3D {b["n"]:.4f}+/-{b["tot"]:.4f}+/-{b.get("sys_deg",0):.4f} vs 0.1667 -> '
          f'{sig(b["n"]-1/6,b.get("comb", b["tot"])):.1f} sigma')
    for tag, num in (('2D', 3), ('3D', 4)):
        Rr = R[tag]
        pk = max(Rr, key=lambda m: Rr[m]['n'])
        e1 = float(np.hypot(Rr[pk].get('comb', Rr[pk]['tot']), Rr[0.0].get('comb', Rr[0.0]['tot'])))
        line = (f'C{num}  {tag} peak {Rr[pk]["n"]:.4f} at mu_g={pk:g}; '
                f'rise {sig(Rr[pk]["n"]-Rr[0.0]["n"], e1):.1f} sigma')
        if 1.0 in Rr:
            e2 = float(np.hypot(Rr[pk].get('comb', Rr[pk]['tot']), Rr[1.0].get('comb', Rr[1.0]['tot'])))
            line += f', fall {sig(Rr[pk]["n"]-Rr[1.0]["n"], e2):.1f} sigma'
        print(line)
        # WINNER'S CURSE. The peak is the max over the sampled frictions, so
        # both its height and its location are selected on noise. This was
        # previously computed only in check_final.py, on ungated quadratic
        # fits, and drifted out of date; it belongs here, on the same fits the
        # constraint uses. The location distribution is the more important
        # output: where it is not concentrated, the peak position is not a
        # result and should not be quoted.
        # Reported over TWO friction sets. The constraint list samples CORE;
        # Fig. 4(a) plots every friction in the sweep, and in 3D the plotted
        # set contains a cell (mu_g = 0.4) larger than the CORE maximum. Sec.
        # IV.B quotes the plotted set, so both must be computed here or that
        # sentence traces to nothing.
        C_all = S[tag][0]
        D_ = S[tag][1]
        R_all = {}
        for m in sorted(C_all):
            ts_ = jammed(C_all[m], D_ + 1)
            if len(ts_) < 4:
                continue
            r_ = nfit.fit_local_sys(C_all[m], T0, fn=nfit.fit_local_adaptive,
                                    restrict_to=ts_, z_min=D_ + 1)
            if r_:
                R_all[m] = float(np.hypot(r_['tot'], r_.get('sys_deg', 0.0))), r_['n']
        for setname, pairs in (('CORE', [(m, Rr[m].get('comb', Rr[m]['tot']),
                                          Rr[m]['n']) for m in sorted(Rr)]),
                               ('plotted', [(m, e_, n_)
                                            for m, (e_, n_) in sorted(R_all.items())])):
            ms = [p[0] for p in pairs]
            es = np.array([p[1] for p in pairs])
            ns = np.array([p[2] for p in pairs])
            rng = np.random.default_rng(3)
            draws = ns[None, :] + rng.normal(0, es, size=(20000, len(ms)))
            am = draws.argmax(axis=1)
            bias = float(np.mean(draws.max(axis=1) - ns[am]))
            loc = np.bincount(am, minlength=len(ms)) / len(am)
            top = ', '.join(f'{ms[i]:g}:{loc[i]:.0%}'
                            for i in np.argsort(-loc) if loc[i] >= 0.05)
            print(f'    selection bias, {setname:<7} ({len(ms)} frictions): '
                  f'height biased high by {bias:+.4f}; argmax at  {top}')
    p2 = max(R['2D'], key=lambda m: R['2D'][m]['n'])
    p3 = max(R['3D'], key=lambda m: R['3D'][m]['n'])
    e2v = R['2D'][p2]['n'] - R['2D'][0.0]['n']
    e3v = R['3D'][p3]['n'] - R['3D'][0.0]['n']
    s2 = float(np.hypot(R['2D'][p2]['tot'], R['2D'][0.0]['tot']))
    s3 = float(np.hypot(R['3D'][p3]['tot'], R['3D'][0.0]['tot']))
    Rr = e3v / e2v
    Rs = abs(Rr) * float(np.hypot(s3 / e3v, s2 / e2v))
    print(f'C5  A_3/A_2 = {Rr:.2f} +/- {Rs:.2f} -> {sig(Rr-1,Rs):.1f} sigma from unity')

    # curvature sign test (constraint 6)
    pos = tot = 0
    line = []
    for tag, (C, D, _) in S.items():
        for m in sorted(C):
            ts = jammed(C[m], D + 1)
            if len(ts) < 4:
                continue
            r = nfit.fit_local(C[m], T0, restrict_to=ts, z_min=D + 1)
            if not r:
                continue
            tot += 1; pos += r['curv'] > 0
            line.append(f'{tag}{m:g}:{"+" if r["curv"]>0 else "-"}')
    from math import comb
    p = 2 * sum(comb(tot, k) for k in range(max(pos, tot - pos), tot + 1)) / 2 ** tot
    print(f'C6  curvature > 0 in {pos}/{tot} cells, sign test p = {min(p,1):.3f}'
          f'   (report claims 17/21, p = 0.0072)')
    print('    ' + ' '.join(line))

    # dimension divergence
    print()
    print('=' * 70)
    print('DIMENSION EFFECT   2D vs 3D at matched mu_g')
    print('=' * 70)
    lo = hi = 0.0
    nlo = nhi = 0
    for m in CORE:
        if m not in R['2D'] or m not in R['3D']:
            continue
        d = R['3D'][m]['n'] - R['2D'][m]['n']
        e = float(np.hypot(R['2D'][m]['tot'], R['3D'][m]['tot']))
        z = d / e
        print(f'   mu_g={m:<5g} 3D-2D = {d:+.4f} +/- {e:.4f}   {z:+.1f} sigma')
        if m <= 0.2:
            lo += z ** 2; nlo += 1
        else:
            hi += z ** 2; nhi += 1
    print(f'\n   mu_g <= 0.2 : chi2/dof = {lo/max(nlo,1):.2f}  (dimensions agree)')
    print(f'   mu_g >  0.2 : chi2/dof = {hi/max(nhi,1):.2f}  (dimensions differ)')
    print('   NOTE: these two numbers are NOT window-controlled -- see below.')

    fn = nfit.fit_local_adaptive if '--adaptive' in sys.argv else nfit.fit_local
    if '--adaptive' in sys.argv:
        print('\n   [--adaptive: polynomial degree chosen per cell by F-test]')
    window_sensitivity(S, K, T0, fn)


def window_sensitivity(S, K, T0, fn=None):
    """The Theta-window systematic that the constraint list does not carry.

    WHY. analyze_iscan3 establishes that with a curved mu(Theta) a quadratic's
    slope at Theta_0 depends on the RANGE fitted, not only on Theta_0, and
    therefore matches the Theta window across the I arms before comparing them.
    That argument does not stop at the I arms. The cells above are fitted over
    their OWN windows -- 2D mu_g = 0 spans 84x, the 2D frictional cells ~20x,
    3D mu_g = 0 spans 173x -- and then compared at one Theta_0.

    Exact matching across the whole table is impossible: the window common to
    all sixteen cells is only 3.2x, which will not support a quadratic. So this
    is pairwise -- for each comparison actually claimed, both cells are refit
    over the range they share, at the SAME Theta_0, so the only thing that
    changes is the fitted range.

    ERRORS HERE ARE DELIBERATELY CONSERVATIVE (stat and the RAW jackknife, not
    the calibrated excess). Restricting the window costs setpoints, which
    raises the jackknife's null expectation and truncates the excess to zero --
    at mu_g = 0.15 and 0.2 in 3D that makes the matched error SMALLER than the
    unmatched one and flatters the comparison. The raw jackknife has no such
    truncation, so the two columns are on the same footing.
    """
    fn = fn or nfit.fit_local

    def rng(tag, m):
        C = S[tag][0]
        ts = K[tag][m]
        # z_min here too: this block's Theta ranges and fits must sit on the
        # same runs as the main table, or the window systematic it reports is
        # measured against a different sample.
        _, byT = nfit.surviving_setpoints(C[m], restrict_to=ts,
                                          z_min=S[tag][1] + 1)
        th = [x['Theta'] for t in ts for x in byT[t] if x['Theta'] > 0]
        return ts, byT, min(th), max(th)

    def pair(m):
        _, _, lo1, hi1 = rng('2D', m)
        _, _, lo2, hi2 = rng('3D', m)
        LO, HI = max(lo1, lo2), min(hi1, hi2)
        th0 = float(np.sqrt(LO * HI))
        out = {}
        for tag in ('2D', '3D'):
            ts, byT, _, _ = rng(tag, m)
            tsm = [t for t in ts
                   if all(LO * 0.999 <= x['Theta'] <= HI * 1.001
                          for x in byT[t] if x['Theta'] > 0)]
            zt = S[tag][1] + 1
            a = nfit.fit_local_sys(S[tag][0][m], th0, fn=fn, restrict_to=ts,
                                   z_min=zt)
            b = (nfit.fit_local_sys(S[tag][0][m], th0, fn=fn, restrict_to=tsm,
                                    z_min=zt)
                 if len(tsm) >= 4 else None)
            out[tag] = (a, b)
        return out, HI / LO

    def cons(r):
        return (float(np.hypot(r['stat'], r['sys_raw']))
                if np.isfinite(r.get('sys_raw', np.nan)) else r['tot'])

    MS = [m for m in CORE if m > 0 and m in K['2D'] and m in K['3D']]
    print()
    print('=' * 78)
    print('WINDOW SYSTEMATIC   the error the constraint list does not carry')
    print('=' * 78)
    print('  A. Each cell refitted over the window it SHARES with its partner,')
    print('     at fixed Theta_0. Only the fitted range changes.')
    print(f'  {"mu_g":>5}{"dim":>5}{"k own":>7}{"k mat":>7}{"n own":>10}'
          f'{"n matched":>11}{"shift":>9}{"published tot":>15}{"shift/tot":>11}')
    for m in MS:
        got, _ = pair(m)
        for tag in ('2D', '3D'):
            a, b = got[tag]
            if not a:
                continue
            if not b:
                print(f'  {m:>5g}{tag:>5}{a["k"]:>7}{"-":>7}   VOID')
                continue
            d = b['n'] - a['n']
            print(f'  {m:>5g}{tag:>5}{a["k"]:>7}{b["k"]:>7}{a["n"]:>10.4f}'
                  f'{b["n"]:>11.4f}{d:>+9.4f}{a["tot"]:>15.4f}'
                  f'{abs(d)/a["tot"]:>10.1f}x')
    print('     The 2D column barely moves: 2D windows are the narrower ones, so')
    print('     2D cells lose no setpoints. The systematic is a 3D systematic.')

    print()
    print('  B. The dimension effect, conservative errors in both columns.')
    print(f'  {"mu_g":>5}{"own windows":>20}{"matched":>20}')
    acc = {'lo': [0.0, 0.0, 0], 'hi': [0.0, 0.0, 0]}
    for m in MS:
        got, span = pair(m)
        cells_, zs = [], []
        for i in (0, 1):
            a, b = got['2D'][i], got['3D'][i]
            if not a or not b:
                cells_.append('VOID'); zs.append(None); continue
            d = b['n'] - a['n']
            e = float(np.hypot(cons(a), cons(b)))
            cells_.append(f'{d:+.4f} {d/e:>5.1f}s'); zs.append(d / e)
        print(f'  {m:>5g}{cells_[0]:>20}{cells_[1]:>20}')
        if zs[0] is not None and zs[1] is not None:
            k = 'lo' if m <= 0.2 else 'hi'
            acc[k][0] += zs[0] ** 2; acc[k][1] += zs[1] ** 2; acc[k][2] += 1
    for k, lab in (('lo', 'mu_g <= 0.2'), ('hi', 'mu_g >  0.2')):
        s, t, n = acc[k]
        if n:
            print(f'    {lab} : chi2/dof   own {s/n:6.2f}   matched {t/n:6.2f}')
    print('  The published reading -- dimensions agree below 0.2 and diverge above --')
    print('  holds only in the "own windows" column. Matched, it inverts.')



if __name__ == '__main__':
    main()
