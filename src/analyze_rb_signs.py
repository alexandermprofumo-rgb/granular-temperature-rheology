"""
Do the channel weights explain the two sign changes near mu_g ~ 0.10?

Two sign changes were found and shown to be consistent with a single
crossing:

    curvature of mu(Theta)      crosses zero at mu_g = 0.103 +/- 0.009
    pressure response dn/dln(k) crosses zero at mu_g = 0.122 +/- 0.025

Both are derivatives of n, and n = C_c + C_n + C_t exactly (§7b), so BOTH
decompose channel by channel. That converts "two crossings coincide" from a
numerical coincidence into a question with a mechanical answer: which channel
changes sign, and does the same one drive both?

TWO IDENTITIES USED, so no new fitting machinery is needed:

  curvature = dn/dlnTheta.  In nfit's convention ln mu = a + b x + c x^2 with
  n = -b and curv = -2c, so curv = d n/d lnTheta exactly. The channel
  contribution is therefore just C_i evaluated at two reference temperatures
  and differenced -- no third derivatives of a_i, which would be hopeless.

  pressure response = dn/dln(kappa), kappa = E/P. With only P = 5 and P = 50
  this is a secant, (C_i(5) - C_i(50))/ln(10). It is centred at the geometric
  mean P ~ 15.8 rather than at 10, which must be stated: the crossing location
  it yields is not required to match the four-pressure value from
  analyze_crossing.py exactly.

Errors come from a leave-one-seed-out jackknife (4 seeds), which is the only
independent replication available at this level.

Usage:  python3 analyze_rb_signs.py
"""
import csv

import numpy as np

import nfit
from analyze_rb import local

PS = (5.0, 50.0)


def load():
    rows = list(csv.DictReader(open('rb_channels.csv')))
    for r in rows:
        for k in ('P', 'mu_g', 'Tgran', 'mu', 'a_c', 'a_n', 'a_t', 'Theta'):
            r[k] = float(r[k])
    return rows


def gated():
    runmap = nfit.load_sweep(
        'sweep_pichi/log.P*',
        r'log\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
        tgran_from=lambda m: float(m.group('T')) * float(m.group('P')) / 10.0)
    cm = {}
    for key, rr in runmap.items():
        d = dict(key)
        cm[(float(d['P']), float(d['mu']))] = rr
    return cm


def channels_at(rows, cm, P, mg, T0, drop_seed=None):
    """C_c, C_n, C_t for one cell at reference temperature T0."""
    rr = cm.get((P, mg))
    if not rr:
        return None
    Ts, _ = nfit.surviving_setpoints(rr)
    keep = {round(t, 12) for t in Ts}
    sel = [r for r in rows
           if r['P'] == P and r['mu_g'] == mg
           and round(r['Tgran'], 12) in keep and r['Theta'] > 0
           and (drop_seed is None or r['seed'] != drop_seed)]
    if len(sel) < 8:
        return None
    x = np.log([r['Theta'] for r in sel])
    x0 = np.log(T0)
    gm = local(x, [r['mu'] for r in sel], x0)
    if gm is None or gm[0] <= 0:
        return None
    out = {}
    for nm in ('a_c', 'a_n', 'a_t'):
        g = local(x, [r[nm] for r in sel], x0)
        if g is None:
            return None
        out[nm] = -g[1] / (2 * gm[0])
    return out


def crossing(mus, vals, errs):
    """Zero crossings located by LOCAL bracketing, not a global straight line.

    A weighted line fitted to the whole mu_g range does two wrong things here:

      * it reports a crossing even when the data never change sign, by
        extrapolating outside the sampled range. C_t at P = 5 is negative at
        every friction measured, yet the line 'crosses' at 0.024.
      * it is a bad model where the channel is non-monotonic. The P = 5
        curvature total is positive at mu_g = 0.05, negative through the middle
        and positive again at 0.5; a straight line through that U reports a
        single crossing that corresponds to nothing.

    So: find consecutive points that actually straddle zero, interpolate
    between them, and report every crossing found. If the data never change
    sign, say so instead of inventing one.
    """
    mus = np.asarray(mus, float); vals = np.asarray(vals, float)
    errs = np.asarray(errs, float)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, np.nan)
    fill = np.nanmax(errs) if np.any(np.isfinite(errs)) else 1.0
    errs = np.where(np.isfinite(errs), errs, fill)
    out = []
    for i in range(len(mus) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0:
            out.append((mus[i], errs[i] * 0))
            continue
        if a * b >= 0:
            continue
        f = a / (a - b)                              # 0..1 along the interval
        x = mus[i] + f * (mus[i + 1] - mus[i])
        # propagate the two endpoint errors through the interpolation
        d = mus[i + 1] - mus[i]
        dfa = -b / (a - b) ** 2
        dfb = a / (a - b) ** 2
        e = abs(d) * float(np.hypot(dfa * errs[i], dfb * errs[i + 1]))
        out.append((x, e))
    return out


def fmt_cross(cr, lo, hi):
    if not cr:
        return f'no sign change in [{lo:g}, {hi:g}]'
    return ' and '.join(f'{x:.4f} +/- {e:.4f}' for x, e in cr)


def main():
    rows = load()
    cm = gated()
    mus = sorted({r['mu_g'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})

    # common reduced-Theta window
    los, his = [], []
    for (P, mg), rr in cm.items():
        Ts, byT = nfit.surviving_setpoints(rr)
        if len(Ts) < 4:
            continue
        th = [x['Theta'] / P for t in Ts for x in byT[t] if x['Theta'] > 0]
        if th:
            los.append(min(th)); his.append(max(th))
    lo, hi = max(los), min(his)
    # two reference temperatures, well inside the common window
    ThLo, ThHi = lo * (hi / lo) ** 0.25, lo * (hi / lo) ** 0.75
    dln = np.log(ThHi / ThLo)
    print(f'common reduced window [{lo:.2e}, {hi:.2e}]')
    print(f'evaluating channels at reduced Theta = {ThLo:.2e} and {ThHi:.2e} '
          f'(dlnTheta = {dln:.3f})\n')

    def chan(P, mg, That, seed=None):
        return channels_at(rows, cm, P, mg, That * P, drop_seed=seed)

    # curvature
    print('=' * 78)
    print('SIGN CHANGE 1 -- curvature = dn/dlnTheta, by channel')
    print('=' * 78)
    print(f'{"P":>5}{"mu_g":>7}  {"dC_c":<18}{"dC_n":<18}{"dC_t":<18}{"total":<18}')
    curv = {}
    for P in PS:
        for mg in mus:
            a = chan(P, mg, ThLo); b = chan(P, mg, ThHi)
            if not (a and b):
                continue
            d = {k: (b[k] - a[k]) / dln for k in a}
            # seed jackknife
            jk = []
            for s in seeds:
                a2 = chan(P, mg, ThLo, s); b2 = chan(P, mg, ThHi, s)
                if a2 and b2:
                    jk.append({k: (b2[k] - a2[k]) / dln for k in a2})
            e = {}
            for k in d:
                if len(jk) >= 3:
                    v = np.array([j[k] for j in jk])
                    e[k] = float(np.sqrt((len(v) - 1) / len(v) * np.sum((v - d[k]) ** 2)))
                else:
                    e[k] = np.nan
            tot = sum(d.values())
            te = float(np.sqrt(np.nansum([e[k] ** 2 for k in e])))
            curv.setdefault(P, []).append((mg, d, e, tot, te))
            print(f'{P:>5g}{mg:>7g}  {d["a_c"]:+.4f}+/-{e["a_c"]:.4f}'
                  f'  {d["a_n"]:+.4f}+/-{e["a_n"]:.4f}'
                  f'  {d["a_t"]:+.4f}+/-{e["a_t"]:.4f}'
                  f'  {tot:+.4f}+/-{te:.4f}')
    print()
    for P in PS:
        if P not in curv or len(curv[P]) < 3:
            continue
        mm = [c[0] for c in curv[P]]
        print(f'   P = {P:g} zero crossings in mu_g:')
        for k, nm in (('a_c', 'C_c'), ('a_n', 'C_n'), ('a_t', 'C_t')):
            z = crossing(mm, [c[1][k] for c in curv[P]], [c[2][k] for c in curv[P]])
            print(f'      {nm}:   {fmt_cross(z, min(mm), max(mm))}')
        z = crossing(mm, [c[3] for c in curv[P]], [c[4] for c in curv[P]])
        print(f'      TOTAL: {fmt_cross(z, min(mm), max(mm))}')
        print()

    # pressure response
    print('=' * 78)
    print('SIGN CHANGE 2 -- dn/dln(kappa), by channel   [secant P=5 vs 50]')
    print('=' * 78)
    dlnk = np.log(10.0)
    print(f'{"mu_g":>7}{"dC_c":>10}{"dC_n":>10}{"dC_t":>10}{"total":>10}')
    pres = []
    for mg in mus:
        a = chan(5.0, mg, np.sqrt(lo * hi)); b = chan(50.0, mg, np.sqrt(lo * hi))
        if not (a and b):
            continue
        d = {k: (a[k] - b[k]) / dlnk for k in a}      # kappa larger at low P
        jk = []
        for s in seeds:
            a2 = chan(5.0, mg, np.sqrt(lo * hi), s); b2 = chan(50.0, mg, np.sqrt(lo * hi), s)
            if a2 and b2:
                jk.append({k: (a2[k] - b2[k]) / dlnk for k in a2})
        e = {}
        for k in d:
            if len(jk) >= 3:
                v = np.array([j[k] for j in jk])
                e[k] = float(np.sqrt((len(v) - 1) / len(v) * np.sum((v - d[k]) ** 2)))
            else:
                e[k] = np.nan
        tot = sum(d.values())
        te = float(np.sqrt(np.nansum([e[k] ** 2 for k in e])))
        pres.append((mg, d, e, tot, te))
        print(f'{mg:>7g}{d["a_c"]:>10.5f}{d["a_n"]:>10.5f}{d["a_t"]:>10.5f}{tot:>10.5f}')
    print()
    if len(pres) >= 3:
        mm = [q[0] for q in pres]
        print('   zero crossings in mu_g:')
        for k, nm in (('a_c', 'C_c'), ('a_n', 'C_n'), ('a_t', 'C_t')):
            z = crossing(mm, [q[1][k] for q in pres], [q[2][k] for q in pres])
            print(f'      {nm}:   {fmt_cross(z, min(mm), max(mm))}')
        z = crossing(mm, [q[3] for q in pres], [q[4] for q in pres])
        print(f'      TOTAL: {fmt_cross(z, min(mm), max(mm))}')
        print()
        print('   NOTE: this secant is centred at the geometric mean P ~ 15.8, not at')
        print('   P = 10, so it need not reproduce the four-pressure crossing exactly.')


if __name__ == '__main__':
    main()
