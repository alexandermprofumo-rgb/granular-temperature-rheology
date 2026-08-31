import numpy as np
import nfit
from math import comb

CORE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
RX = r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$'


def by_mu(pat):
    out = {}
    for k, v in nfit.load_sweep(pat, RX).items():
        out.setdefault(float(dict(k)['mu']), []).extend(v)
    return out


m2, d3 = by_mu('sweep_matched2d/log.mu*'), by_mu('sweep_run_3d/log.mu*')
pool = [m2[m] for m in CORE if m in m2] + [d3[m] for m in CORE if m in d3]
T0, _ = nfit.common_theta0(pool)

# ------------------------------------------------- which gate is binding?
print('=' * 78)
print('WHICH GATE ACTUALLY REMOVES SETPOINTS?')
print('=' * 78)
for tag, C in (('2D', m2), ('3D', d3)):
    for mg in CORE:
        if mg not in C:
            continue
        runs = C[mg]
        allT = sorted({round(r['Tgran'], 12) for r in runs})
        Ts, byT = nfit.surviving_setpoints(runs)
        dropped = [t for t in allT if t not in Ts]
        why = []
        for t in dropped:
            g = byT[t]
            Im = np.mean([r['I'] for r in g])
            med = np.median([np.mean([r['I'] for r in byT[u]]) for u in allT])
            th = np.mean([r['Theta'] for r in g])
            ps = (np.mean([r['Ps'] for r in g]) / np.mean([r['Pm'] for r in g])
                  if 'Ps' in g[0] else np.nan)
            tags = []
            if np.isfinite(ps) and ps > nfit.P_TOL:
                tags.append(f'baro {ps:.2f}')
            if abs(Im / med - 1) > nfit.I_TOL:
                tags.append(f'I {Im/med-1:+.2%}')
            if th / t > 1.0:
                tags.append(f'Theta/T {th/t:.2f}')
            why.append(f'{t:g}[{",".join(tags) or "?"}]')
        print(f'   {tag} mu={mg:<5g} {len(allT)} setpoints -> {len(Ts)} kept; '
              f'dropped {" ".join(why) if why else "(none)"}')

# ------------------------------------------- curvature sign, all cells
print()
print('=' * 78)
print('SECTION 4 / CONSTRAINT 6: does the local slope rise with Theta?')
print('sign of the fitted curvature (= d n / d ln Theta), every cell')
print('=' * 78)
pos = tot = 0
line = []
for tag, C in (('2D', m2), ('3D', d3)):
    for mg in sorted(C):
        r = nfit.fit_local(C[mg], T0)
        if not r:
            continue
        tot += 1; pos += r['curv'] > 0
        line.append(f'{tag}{mg:g}:{r["curv"]:+.3f}')
p = 2 * sum(comb(tot, k) for k in range(max(pos, tot - pos), tot + 1)) / 2 ** tot
print('   ' + ', '.join(line))
print(f'   {pos} of {tot} cells have curvature > 0  ->  two-sided sign test '
      f'p = {min(p,1.0):.3f}')
print('   (report section 4 claims 17 of 21, p = 0.0072)')

# --------------------------------------- winner's curse on the peak
print()
print('=' * 78)
print("PEAK SELECTION BIAS: constraints_audited picks the max of 8 noisy cells")
print('=' * 78)
R = {}
for tag, C in (('2D', m2), ('3D', d3)):
    R[tag] = {m: nfit.fit_local_sys(C[m], T0) for m in CORE if m in C}
rng = np.random.default_rng(3)
for tag in ('2D', '3D'):
    ns = np.array([R[tag][m]['n'] for m in sorted(R[tag])])
    es = np.array([R[tag][m]['tot'] for m in sorted(R[tag])])
    pk = ns.argmax()
    # bootstrap: how much does 'max over 8' overshoot the true max?
    over = []
    for _ in range(20000):
        d = ns + rng.normal(0, es)
        over.append(d.max() - ns[d.argmax()])
    print(f'   {tag}: quoted peak {ns.max():.4f} at mu_g='
          f'{sorted(R[tag])[pk]:g};  expected upward bias of "max of 8" '
          f'given these errors: {np.mean(over):+.4f}')
    loc = [sorted(R[tag])[(ns + rng.normal(0, es)).argmax()] for _ in range(5000)]
    from collections import Counter
    c = Counter(loc)
    print('        peak location under resampling: '
          + ', '.join(f'{k:g}:{v/5000:.0%}' for k, v in sorted(c.items())))

# ------------------------------- constraint list with a calibrated systematic
print()
print('=' * 78)
print('CONSTRAINT LIST WITH THE JACKKNIFE CALIBRATED AGAINST ITS OWN NULL')
print('  sys_cal = sqrt(max(sys^2 - sys_null^2, 0));  tot_cal = hypot(stat, sys_cal)')
print('=' * 78)
NREP = 200


def null_sys(runs, rng):
    Ts, byT = nfit.surviving_setpoints(runs)
    sel = [x for t in Ts for x in byT[t]]
    u = np.log(np.array([x['Theta'] for x in sel]) / T0)
    y = np.log(np.array([x['mu'] for x in sel]))
    A = np.vstack([np.ones_like(u), u, u ** 2]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    s = (y - A @ c).std(ddof=3)
    o = []
    for _ in range(NREP):
        fake = [dict(x, mu=float(np.exp((A @ c)[i] + rng.normal(0, s))))
                for i, x in enumerate(sel)]
        f = nfit.fit_local_sys(fake, T0)
        if f and np.isfinite(f['sys']):
            o.append(f['sys'])
    return float(np.mean(o))


rng = np.random.default_rng(0)
CAL = {}
print(f'{"cell":<11}{"n":>9}{"tot (report)":>14}{"tot (calibrated)":>18}')
for tag in ('2D', '3D'):
    for mg in sorted(R[tag]):
        r = R[tag][mg]
        nl = null_sys((m2 if tag == '2D' else d3)[mg], rng)
        exc = float(np.sqrt(max(r['sys'] ** 2 - nl ** 2, 0.0)))
        t = float(np.hypot(r['stat'], exc))
        CAL[(tag, mg)] = (r['n'], t)
        print(f'{tag+" "+format(mg,"g"):<11}{r["n"]:>+9.4f}{r["tot"]:>14.4f}'
              f'{t:>18.4f}')


def sig(d, e):
    return abs(d) / e


print()
n2, e2 = CAL[('2D', 0.0)]; n3, e3 = CAL[('3D', 0.0)]
print(f'   C2 geometric theory: 2D {n2:.4f}+/-{e2:.4f} vs 0.25 -> '
      f'{sig(n2-0.25, e2):.1f} sigma  (report 6.8)')
print(f'                        3D {n3:.4f}+/-{e3:.4f} vs 0.1667 -> '
      f'{sig(n3-1/6, e3):.1f} sigma  (report 5.0)')
for tag in ('2D', '3D'):
    ms = sorted(R[tag])
    pk = max(ms, key=lambda m: CAL[(tag, m)][0])
    pn, pe = CAL[(tag, pk)]
    bn, be = CAL[(tag, 0.0)]; tn, te = CAL[(tag, 1.0)]
    print(f'   C3/4 {tag}: peak {pn:.4f} at mu={pk:g};  rise '
          f'{sig(pn-bn, np.hypot(pe,be)):.1f} sigma, fall '
          f'{sig(pn-tn, np.hypot(pe,te)):.1f} sigma')
e2v = CAL[('2D', max(sorted(R['2D']), key=lambda m: CAL[('2D', m)][0]))][0] - CAL[('2D', 0.0)][0]
e3v = CAL[('3D', max(sorted(R['3D']), key=lambda m: CAL[('3D', m)][0]))][0] - CAL[('3D', 0.0)][0]
s2 = float(np.hypot(CAL[('2D', 0.3)][1], CAL[('2D', 0.0)][1]))
s3 = float(np.hypot(CAL[('3D', 0.3)][1], CAL[('3D', 0.0)][1]))
Rr = e3v / e2v
Rs = abs(Rr) * float(np.hypot(s3 / e3v, s2 / e2v))
print(f'   C5 A_3/A_2 = {Rr:.2f} +/- {Rs:.2f}  -> {sig(Rr-1, Rs):.1f} sigma from 1'
      f'   (report table 1.41 +/- 0.41; report text 2.38 +/- 0.95)')
