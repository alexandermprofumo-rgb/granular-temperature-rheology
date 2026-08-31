"""The 3D stiffness lever: is chi a state function for n, and for the channels?

WHY THIS EXISTS. ANALYSIS_PROTOCOL section 6 records a standing limitation:
"Constraints 7-10 and all lever results are 2D only." The two lever results
that matter most are

  constraint 8   chi is NOT a state function for n.
  section 7      whether n_c and n_n ARE functions of chi even though n is not.

This docstring used to quote 2D numbers (chi2/dof 39.4 and 15.7 sigma for
constraint 8; 1.1 and 0.8 "passes" for section 7) that had been transcribed by
hand from a report and were never recomputed after the 2026-08-20
re-extraction. The 2026-08-25 audit could not reproduce them under ANY frame
policy, fabric weighting, gate or Theta_0 -- the best value obtainable over 84
configurations was 2.12 for n_c, still failing the < 2 criterion. The 2D side
is therefore no longer written down here at all: it is read at run time from
channel_state_2d.json, which analyze_channel_state.py writes. No number in this
project should reach a table by transcription again.

sweep_lev_E3d (3 stiffnesses x 3 frictions x 12
temperatures x 2 seeds = 216 runs) is the 3D counterpart of the 2D stiffness
lever, so both can now be repeated in 3D. As of the audit the 2D side also
fails on all 32 pairs, so this is no longer a replication test of a positive
result -- it is the 3D half of a uniform null across 64 (channel, candidate)
pairs.

THE TEST IS THE PRE-REGISTERED ONE, unchanged: pool the friction lever and the
stiffness lever, fit ONE curve Y(X) quadratic in ln X, and require BOTH

    (a) chi2/dof < 2 for the single curve, and
    (b) no residual trend against ln E at >= 2 sigma.

THREE THINGS DIFFER FROM 2D, all forced by the geometry:

 1. The identity is  mu = D*A_c + D*A_n + mu_t  (D = 3), not (a_c+a_n+a_t)/2,
    so the channel derivatives carry D and mu_t rather than a_t.
 2. The jamming gate is Z >= D+1 = 4, not 3.
 3. Frames are selected by the CORRECTED policy -- the whole measurement half.
    analyze_channels_3d_steady.py uses keep_last=8, which is unbiased but noisy;
    in 2D that choice moved the three-channel consistency gate from 0.7 to 2.6
    sigma. Earlier 3D numbers still use keep_last=8, so
    this script is NOT a drop-in replacement for them -- it is the stiffness
    lever, extracted under the current policy.

Usage:  python3 analyze_lev_E3d.py [--recache]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import nfit

NCOL, D = 14, 3
NGRAIN = 4000
ZMIN = 4.0                      # D+1 in 3D, not the 2D value of 3
MOB = 0.99
CACHE = 'chan_lev_E3d.csv'
CACHE_F = 'chan_steady3d_half.csv'
CHANNELS = ('n_c', 'n_n', 'C_t', 'n_tot')
CANDS = ('Z', 'chi', 'cv_f', 'pr_f', 'f_mean', 'A_c', 'A_n', 'mu_t')

SRC = {
    CACHE: ('sweep_lev_E3d', r'dump\.contacts\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
            r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ('E',)),
    CACHE_F: ('sweep_steady3d', r'dump\.contacts\.mu(?P<mu>[0-9.]+)'
              r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ()),
}


def tail_frames(path, ncol=NCOL):
    """Every MEASUREMENT-WINDOW frame, without parsing the equilibration half.

    nequil == nmeas in every campaign, so the measurement window is exactly the
    second half of the dump. Seek to just before it rather than reading the
    whole file: correct AND several times faster. Same policy as
    extract_channels.tail_frames, which is what the current 2D numbers use.
    """
    size = os.path.getsize(path)
    with open(path) as f:
        head = f.read(4096)
    m = re.search(r'ITEM: NUMBER OF ENTRIES\s*\n\s*(\d+)', head)
    if not m:
        return []
    approx = int(m.group(1)) * 17 + 300          # bytes/frame, generous for 3D
    nfr = max(1, size // max(approx, 1))
    keep = max(4, int(nfr // 2))
    start = max(0, size - approx * (keep + 1))
    out = []
    with open(path) as f:
        f.seek(start)
        f.readline()                              # discard the partial line
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                try:
                    k = int(f.readline())
                except (TypeError, ValueError):
                    break
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                rows = []
                for _ in range(k):
                    t = f.readline().split()
                    if len(t) >= ncol:
                        rows.append([float(x) for x in t[:ncol]])
                if rows:
                    out.append(np.array(rows))
    return out[-keep:] if out else []


def channels(path, mu_g):
    """3D exact decomposition, plus chi and the force-distribution moments.

    mu = D*A_c + D*A_n + mu_t exactly; the residual of that identity is
    returned so it can be checked rather than assumed.
    """
    acc = []
    for a in tail_frames(path):
        d, fnv, ftv = a[:, 4:7], a[:, 7:10], a[:, 10:13]
        ftm = a[:, 13]
        L = np.linalg.norm(d, axis=1)
        fnm = np.linalg.norm(fnv, axis=1)
        live = (fnm > 0) & (L > 0)
        if live.sum() < 200:
            continue
        d, fnv, ftv, L = d[live], fnv[live], ftv[live], L[live]
        ftm, fnm = ftm[live], fnm[live]
        nh = d / L[:, None]
        Sn = np.einsum('ci,cj->ij', fnv, d)
        St = np.einsum('ci,cj->ij', ftv, d)
        Sn = .5 * (Sn + Sn.T); St = .5 * (St + St.T)
        S = Sn + St
        tr = np.trace(S)
        if tr <= 0:
            continue
        P = tr / D
        dev = S - np.eye(D) * tr / D
        nrm = np.sqrt(np.sum(dev * dev))
        if nrm <= 0:
            continue
        sh = dev / nrm
        mu = float(np.sum(dev * sh) / P)
        mu_t = float(np.sum((St - np.eye(D) * np.trace(St) / D) * sh) / P)
        fs = np.einsum('ij,ij->i', fnv, nh)
        v = L / L.sum()
        wt = L * fs
        if wt.sum() <= 0:
            continue
        w = wt / wt.sum()
        F = np.einsum('c,ci,cj->ij', v, nh, nh)
        G = np.einsum('c,ci,cj->ij', w, nh, nh)
        Fd = F - np.eye(D) * np.trace(F) / D
        Gd = G - np.eye(D) * np.trace(G) / D
        A_c = float(np.sum(Fd * sh))
        A_n = float(np.sum((Gd - Fd) * sh))
        fm = float(fs.mean())
        chi = float((ftm / (mu_g * fnm) >= MOB).mean()) if mu_g > 0 else 0.0
        acc.append((mu, mu_t, A_c, A_n, chi, fm,
                    float(fs.std() / max(fm, 1e-30)),
                    float((fs.sum() ** 2) / max(len(fs) * (fs ** 2).sum(), 1e-30)),
                    float(live.sum()),
                    (D * A_c + D * A_n + mu_t - mu) / max(abs(mu), 1e-30)))
    return np.mean(acc, axis=0) if acc else None


def build(cache, recache=False):
    if os.path.exists(cache) and not recache:
        rows = list(csv.DictReader(open(cache)))
        for r in rows:
            for k in r:
                if k != 'seed':
                    r[k] = float(r[k])
        return rows, None
    sweep, rx_s, extra = SRC[cache]
    rx = re.compile(rx_s)
    rows, worst = [], 0.0
    paths = sorted(glob.glob(f'{sweep}/dump.contacts.*'))
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        g = m.groupdict()
        lab = os.path.basename(p).replace('dump.contacts.', '')
        q = nfit.parse_log(f'{sweep}/log.{lab}')
        if not q:
            continue
        c = channels(p, float(g['mu']))
        if c is None:
            continue
        worst = max(worst, abs(c[9]))
        row = dict(mu_g=float(g['mu']), Tgran=float(g['T']), seed=g['s'],
                   Theta=q['Theta'], mu=c[0], mu_t=c[1], A_c=c[2], A_n=c[3],
                   chi=c[4], f_mean=c[5], cv_f=c[6], pr_f=c[7], ncon=c[8])
        for k in extra:
            row[k] = float(g[k])
        rows.append(row)
        if (i + 1) % 40 == 0:
            print(f'   {sweep} {i+1}/{len(paths)}', file=sys.stderr)
    if rows:
        with open(cache, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows, worst


def loc(x, y, x0, deg=2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < deg + 2:
        return None
    A = np.vander(x - x0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    cov = (float(r @ r) / max(len(x) - deg - 1, 1)) * np.linalg.inv(A.T @ A)
    return float(c[0]), float(c[1]), float(np.sqrt(cov[1, 1]))


def gated_setpoints(sweep, rx_s, keyfn, zmin=ZMIN):
    """Per cell, the setpoints surviving all four protocol gates (section 2).

    This script previously applied none of nfit's three gates and applied the
    jamming gate on the cell-MEAN Z rather than per setpoint. Both are fixed
    here; the verdicts do not change, but the numbers are now on the same
    policy as the constraint list.
    """
    import re as _re
    out = {}
    for k, runs in nfit.load_sweep(f'{sweep}/log.*',
                                   rx_s.replace('dump\\.contacts\\.', 'log\\.')).items():
        Ts, byT = nfit.surviving_setpoints(runs, z_min=zmin)
        keep = set()
        for t in Ts:
            z = [x['Z'] for x in byT[t] if 'Z' in x]
            if not z or np.mean(z) >= zmin:
                keep.add(round(t, 12))
        out[keyfn(dict(k))] = keep
    return out


def cells_from(rows, keyfn, T0, gates=None):
    grp = {}
    for r in rows:
        if gates is not None:
            g = gates.get(keyfn(r))
            if g is None or round(r['Tgran'], 12) not in g:
                continue
        if r['Theta'] > 0:
            r['Z'] = 2.0 * r['ncon'] / NGRAIN
            grp.setdefault(keyfn(r), []).append(r)
    out = {}
    for k, rs in grp.items():
        if len(rs) < 8:
            continue
        x = np.log([r['Theta'] for r in rs]); x0 = np.log(T0)
        gm = loc(x, np.log([max(r['mu'], 1e-30) for r in rs]), x0)
        if gm is None:
            continue
        mu0 = float(np.exp(gm[0]))
        d, ok = {}, True
        for nm, key in (('n_c', 'A_c'), ('n_n', 'A_n')):
            v = np.array([r[key] for r in rs])
            if (v <= 0).any():
                ok = False; break
            g = loc(x, np.log(v), x0)
            d[nm], d[nm + '_e'] = -g[1], g[2]
        if not ok:
            continue
        g = loc(x, [r['mu_t'] for r in rs], x0)
        d['C_t'], d['C_t_e'] = -g[1] / mu0, g[2] / mu0
        d['n_tot'], d['n_tot_e'] = -gm[1], gm[2]
        for c in CANDS:
            d[c] = float(np.mean([r[c] for r in rs]))
        d['Zbar'] = float(np.mean([2.0 * r['ncon'] / NGRAIN for r in rs]))
        out[k] = d
    return out


def main():
    rc = '--recache' in sys.argv
    st, w1 = build(CACHE, rc)
    fr, w2 = build(CACHE_F, rc)
    print('=' * 74)
    print('THE 3D STIFFNESS LEVER   --   constraint 8 and section 7, in 3D')
    print('=' * 74)
    print(f'  stiffness lever  {len(st)} runs from sweep_lev_E3d')
    print(f'  friction lever   {len(fr)} runs from sweep_steady3d')
    for nm, w in (('stiffness', w1), ('friction', w2)):
        if w is not None:
            print(f'  identity mu = D*A_c + D*A_n + mu_t, worst {nm} residual: '
                  f'{w:.2e}')
    allth = [r['Theta'] for r in st + fr if r['Theta'] > 0]
    T0 = float(np.exp(np.mean(np.log(allth))))
    print(f'  common Theta_0 = {T0:.3e}\n')

    # DE-DUPLICATE. The E = 1e5 arm of sweep_lev_E3d is bit-identical to
    # sweep_steady3d at the same frictions -- all 72 runs, same md5 on the
    # dumps: LAMMPS is deterministic and the inputs matched exactly, so the
    # driver recomputed runs that already existed. Pooling both would enter
    # three cells twice and double their weight in the collapse fit. The
    # friction lever keeps them (it has ten frictions, not three).
    dup = {(r['mu_g'], r['Tgran'], r['seed']) for r in fr}
    nst = len(st)
    st = [r for r in st
          if not (abs(r['E'] - 1.0e5) < 1 and
                  (r['mu_g'], r['Tgran'], r['seed']) in dup)]
    print(f'  de-duplicated: {nst - len(st)} of {nst} stiffness runs were exact '
          f'copies of\n  friction-lever runs; the lever is really E = 1e4 and '
          f'1e6 against 1e5.\n')

    gA = gated_setpoints(SRC[CACHE_F][0], SRC[CACHE_F][1],
                         lambda d: ('fric', float(d['mu'])))
    gB = gated_setpoints(SRC[CACHE][0], SRC[CACHE][1],
                         lambda d: ('stiff', float(d['E']), float(d['mu'])))
    A = cells_from(fr, lambda r: ('fric', r['mu_g']), T0, gA)
    B = cells_from(st, lambda r: ('stiff', r['E'], r['mu_g']), T0, gB)
    A = {k: v for k, v in A.items() if v['Zbar'] >= ZMIN}
    B = {k: v for k, v in B.items() if v['Zbar'] >= ZMIN}
    print(f'  friction lever: {len(A)} cells;  stiffness lever: {len(B)} cells'
          f'   (all four protocol gates, per setpoint; jamming Z >= {ZMIN})\n')
    if not B:
        print('  no stiffness cell survives the gate -- nothing to test.')
        return

    print(f'  {"channel":<8}{"candidate":<10}{"chi2/dof":>10}{"trend vs lnE":>15}'
          f'  verdict')
    print('  ' + '-' * 56)
    nres, passes = 0, []
    for ch in CHANNELS:
        for cand in CANDS:
            pts = []
            for src, dd in (('f', A), ('s', B)):
                for k, v in dd.items():
                    lev = 1.0e5 if src == 'f' else k[1]
                    if v[cand] > 0 and np.isfinite(v[ch]) and v[ch + '_e'] > 0:
                        pts.append((v[cand], v[ch], v[ch + '_e'], lev))
            if len(pts) < 8:
                continue
            x = np.log([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            e = np.array([p[2] for p in pts]); lE = np.log([p[3] for p in pts])
            if lE.std() == 0:
                continue
            M = np.vstack([np.ones_like(x), x, x ** 2]).T
            W = np.diag(1 / e ** 2)
            c = np.linalg.inv(M.T @ W @ M) @ M.T @ W @ y
            res = y - M @ c
            chi2 = float(((res / e) ** 2).sum()) / max(len(x) - 3, 1)
            Ml = np.vstack([np.ones_like(lE), lE]).T
            cov = np.linalg.inv(Ml.T @ W @ Ml)
            cl = cov @ Ml.T @ W @ res
            # ANALYSIS_PROTOCOL section 4: inflate by sqrt(chi2/dof) where the
            # single-curve fit has already overshot its own errors. Without this
            # the 3D table reads n_c 3.4s / n_n 6.8s / C_t 2.9s; with it, no
            # trend anywhere in 3D is significant and the failures are chi2
            # failures only -- which is the honest description.
            tz = (abs(cl[1]) / np.sqrt(cov[1, 1])) / np.sqrt(max(chi2, 1.0))
            nres += 1
            ok = chi2 < 2.0 and tz < 2.0
            if ok:
                passes.append((ch, cand, chi2, tz))
            print(f'  {ch:<8}{cand:<10}{chi2:>10.1f}{tz:>13.1f}s  '
                  f'{"** PASSES **" if ok else "fails"}   [{len(pts)} cells]')
    print(f'\n  {nres} (channel, candidate) pairs tested -- at 2 sigma you would '
          f'expect ~{0.05*nres:.0f}\n  false passes by chance.')
    print('\n  THE PRE-REGISTERED COMPARISON (2D -> 3D), 2D read from')
    print('  channel_state_2d.json -- run analyze_channel_state.py first:')
    try:
        import json
        two = json.load(open('channel_state_2d.json'))['pairs']
    except (OSError, ValueError, KeyError):
        two = None
        print('   [channel_state_2d.json missing; 2D column unavailable]')
    for ch in CHANNELS:
        got = [p for p in passes if p[0] == ch and p[1] == 'chi']
        lab = '(unavailable)'
        if two and f'{ch}|chi' in two:
            c2, t2, ok2 = two[f'{ch}|chi']
            lab = (f'2D: chi2 {c2:.1f}, trend {t2:.1f}s  '
                   f'{"PASSES" if ok2 else "fails"}')
        print(f'   {ch:<6} vs chi   {lab:<34} -> 3D: '
              f'{"PASSES" if got else "fails"}')


if __name__ == '__main__':
    main()
