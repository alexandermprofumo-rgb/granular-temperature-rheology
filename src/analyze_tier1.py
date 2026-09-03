"""
Tier 1 analysis: force-resolved contact statistics from the sweep produced by
run_tracking_forces_sweep.sh.

Computes, per (mu_g, Theta, seed):
  Z          mean contacts per grain (active contacts only)
  chi        Coulomb-mobilised fraction, |f_t| >= (1-tol) mu_g f_n
  Z_c(chi)   generalised isostatic coordination (see note below)
  dZ_eff     Z - Z_c(chi), the soft-mode control parameter
  a_c        unweighted deviatoric fabric anisotropy  (from F_ij = <n_i n_j>)
  a_c_w      FORCE-WEIGHTED fabric anisotropy         (F^w_ij ~ sum f_n n_i n_j)
  a_c_str/wk fabric anisotropy of the strong / weak sub-networks
  cv_f       coefficient of variation of the normal-force distribution
  pr_f       participation ratio of f_n (1/N_eff; small => few contacts carry
             the load)

It then (i) fits the Theta-scaling exponent of each fabric measure per mu_g
and compares against the macroscopic n(mu_g) from results.csv, and (ii)
reports n against dZ_eff for the collapse test.

GENERALISED ISOSTATICITY -- the one modelling choice here
A contact sitting at its Coulomb threshold slides, so it constrains only the
normal direction; a stuck contact constrains tangentially too. Z_c therefore
interpolates between the fully-frictional and frictionless limits:
    chi = 0  (nothing sliding)  -> Z_c = D + 1     (3 in 2D)
    chi = 1  (all sliding)      -> Z_c = 2D        (4 in 2D)
We use the standard linear interpolation Z_c(chi) = (D+1) + chi (D-1), which
matches both limits. Note that naive constraint counting instead gives
Z_c = 2D(D+1) / (2D - chi(D-1)) ... which reproduces chi=0 but NOT chi=1,
because when every contact slides the grain rotations stop being rigidity
degrees of freedom at all. That endpoint subtlety is why the linear form is
the conventional choice. Both are implemented; --zc-model selects.

At mu_g = 0 there is no tangential force whatsoever, so every contact is
trivially at its (zero) threshold and chi = 1 by definition, correctly giving
the frictionless isostatic value.

Usage:
    python3 analyze_tier1.py --dir sweep_tracking_forces --out tier1_results.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import numpy as np

# Column layouts of dump.contacts.<label>:
#   2D (11): index id1 id2 dist dx dy    fx fy    ftx fty     ftmag
#   3D (14): index id1 id2 dist dx dy dz fx fy fz ftx fty ftz ftmag
NCOL = {2: 11, 3: 14}
# Grain count, used to turn a contact count into a coordination number
# (Z = 2*Ncontacts/N). This MUST match the N the sweep was actually run at.
# The original tracking sweeps used 2000; sweep_tracking_matched2d uses 4000
# so that a_c is comparable to the N=4000 production macroscopic runs. Getting
# it wrong silently rescales Z and hence dZ_eff by N_actual/N_GRAINS -- it
# showed up as a 2D coordination of 8.1, twice the calibrated value of 4.04.
N_GRAINS = 2000


def zc_of_chi(chi, model='linear', D=2):
    if model == 'linear':
        return (D + 1) + chi * (D - 1)
    if model == 'counting':
        return 2 * D * (D + 1) / (2 * D - chi * (D - 1))
    raise ValueError(model)


def iter_frames(path, ncol):
    """Yield (arr, box) per frame. Streams by splitting on the ENTRIES header."""
    with open(path) as f:
        txt = f.read()
    blocks = txt.split('ITEM: TIMESTEP')
    for blk in blocks[1:]:
        mbox = re.search(r'ITEM: BOX BOUNDS[^\n]*\n(.*?)ITEM: ENTRIES[^\n]*\n',
                         blk, re.DOTALL)
        if not mbox:
            continue
        body = blk[mbox.end():]
        if not body.strip():
            continue
        try:
            arr = np.array(body.split(), dtype=float)
        except ValueError:
            continue
        if arr.size < ncol or arr.size % ncol:
            arr = arr[:arr.size - (arr.size % ncol)]
            if arr.size == 0:
                continue
        yield arr.reshape(-1, ncol)


def fabric_anisotropy(nv, w=None):
    """Deviatoric invariant of F_ij = <w n_i n_j>/<w>, for nv shape (M, D).
    2D: a_c = 2 sqrt(((Fxx-Fyy)/2)^2 + Fxy^2)   (matches the methods report)
    3D: a_c = sqrt(3/2 sum_ij (F_ij - delta_ij/3)^2)      (likewise)"""
    if len(nv) == 0:
        return np.nan
    w = np.ones(len(nv)) if w is None else w
    W = w.sum()
    if W <= 0:
        return np.nan
    F = np.einsum('i,ij,ik->jk', w, nv, nv) / W
    if nv.shape[1] == 2:
        return 2 * np.sqrt(((F[0, 0] - F[1, 1]) / 2) ** 2 + F[0, 1] ** 2)
    Dv = F - np.eye(3) / 3.0
    return float(np.sqrt(1.5 * np.sum(Dv * Dv)))


def analyse_run(path, mu_g, dim=2, skip_frac=0.2, tol=0.01, n_grains=N_GRAINS):
    frames = list(iter_frames(path, NCOL[dim]))
    if len(frames) < 3:
        return None
    frames = frames[int(len(frames) * skip_frac):]

    acc = {k: [] for k in ('Z', 'chi', 'a_c', 'a_c_w', 'a_c_str', 'a_c_wk',
                           'cv_f', 'pr_f', 'f_mean')}
    for arr in frames:
        dist = arr[:, 3]
        d = arr[:, 4:4 + dim]
        f = arr[:, 4 + dim:4 + 2 * dim]
        ftm = arr[:, -1]
        fn = np.linalg.norm(f, axis=1)
        act = (fn > 1e-12) & (dist > 0)
        if act.sum() < 10:
            continue
        dist, d, fn, ftm = dist[act], d[act], fn[act], ftm[act]
        nv = d / dist[:, None]

        acc['Z'].append(2.0 * len(fn) / n_grains)
        if mu_g <= 0:
            acc['chi'].append(1.0)          # frictionless: no tangential force
        else:
            acc['chi'].append(float(np.mean(ftm >= (1 - tol) * mu_g * fn)))

        acc['a_c'].append(fabric_anisotropy(nv))
        acc['a_c_w'].append(fabric_anisotropy(nv, w=fn))
        strong = fn > fn.mean()
        acc['a_c_str'].append(fabric_anisotropy(nv[strong]))
        acc['a_c_wk'].append(fabric_anisotropy(nv[~strong]))

        acc['cv_f'].append(fn.std() / fn.mean())
        acc['pr_f'].append(np.sum(fn) ** 2 / (len(fn) * np.sum(fn ** 2)))
        acc['f_mean'].append(fn.mean())

    if not acc['Z']:
        return None
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def parse_theta(logpath):
    try:
        lines = open(logpath).readlines()
    except FileNotFoundError:
        return None
    start = next((i for i, l in enumerate(lines)
                  if 'BEGIN CONTACT TRACKING' in l), None)
    if start is None:
        return None
    rows = []
    for l in lines[start:]:
        t = l.split()
        if len(t) == 3:
            try:
                rows.append([float(x) for x in t])
            except ValueError:
                pass
    if not rows:
        return None
    a = np.array(rows)
    a = a[int(len(a) * 0.3):]
    return float(np.mean(a[:, 2])) if len(a) else None


def fit_exponent(theta, y, sign=+1.0):
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(theta) & np.isfinite(y) & (theta > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, yy = np.log(theta[m]), np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ coef
    s2 = np.sum(resid ** 2) / max(len(x) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return sign * coef[0], np.sqrt(cov[0, 0])


def macroscopic_n(results_csv, core=None):
    """core = set of Tgran values to fit over. The 2D published fits use the
    core window {5e-4, 2e-3, 6e-3, 1.5e-2}; the 3D fits use the full tested
    range (see the methods report), so pass core=None for 3D."""
    rows = list(csv.DictReader(open(results_csv)))
    for r in rows:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    if core:
        rows = [r for r in rows if r['Tgran'] in core]
    out = {}
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            continue
        n, e = fit_exponent([r['Theta'] for r in sub], [r['mu'] for r in sub],
                            sign=-1.0)
        out[mg] = (n, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sweep_tracking_forces')
    ap.add_argument('--out', default='tier1_results.csv')
    ap.add_argument('--macro', default='results.csv')
    ap.add_argument('--zc-model', default='linear', choices=['linear', 'counting'])
    ap.add_argument('--dim', type=int, default=2, choices=[2, 3])
    ap.add_argument('--n-grains', type=int, default=N_GRAINS,
                    help='grain count of the sweep being analysed (Z = 2*Ncontacts/N)')
    args = ap.parse_args()

    pat = re.compile(r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$')
    rows = []
    files = sorted(glob.glob(os.path.join(args.dir, 'dump.contacts.*')))
    if not files:
        print(f'No per-contact dumps under {args.dir}.\n'
              'This script needs tier-3 data, the raw per-contact dumps\n'
              '(~198 GB), which are not distributed. See "Data, and what you\n'
              'need for what" in the README. Nothing in the figure or\n'
              'constraint pipeline depends on this script.')
        return 1
    for i, path in enumerate(files, 1):
        label = os.path.basename(path).split('dump.contacts.', 1)[1]
        m = pat.search(label)
        if not m:
            continue
        mu_g, Tg, seed = float(m.group(1)), float(m.group(2)), int(m.group(3))
        res = analyse_run(path, mu_g, dim=args.dim, n_grains=args.n_grains)
        if res is None:
            print(f'  skip (too few frames): {label}', file=sys.stderr)
            continue
        theta = parse_theta(os.path.join(args.dir, f'log.{label}'))
        res.update(label=label, mu_g=mu_g, Tgran=Tg, seed=seed,
                   Theta=theta if theta else Tg)
        res['Z_c'] = zc_of_chi(res['chi'], args.zc_model, D=args.dim)
        res['dZ_eff'] = res['Z'] - res['Z_c']
        rows.append(res)
        print(f'[{i}/{len(files)}] {label}: Z={res["Z"]:.2f} chi={res["chi"]:.3f} '
              f'dZ_eff={res["dZ_eff"]:+.3f} a_c={res["a_c"]:.4f} '
              f'a_c^w={res["a_c_w"]:.4f}')

    if not rows:
        print('nothing parsed')
        return 1
    fields = sorted({k for r in rows for k in r})
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows to {args.out}')

    core2d = {0.0005, 0.002, 0.006, 0.015}
    nmac = macroscopic_n(args.macro, core2d if args.dim == 2 else None)

    print('\n=== Theta-scaling exponents of each fabric measure vs macroscopic n ===')
    print(f'{"mu_g":>7} {"n_macro":>16} {"unweighted a_c":>16} '
          f'{"FORCE-WTD a_c^w":>17} {"strong":>14} {"weak":>14}')
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        th = [r['Theta'] for r in sub]
        def f(key):
            v, e = fit_exponent(th, [r[key] for r in sub], sign=-1.0)
            return f'{v:+.3f}+/-{e:.3f}'
        nm = nmac.get(mg)
        nms = f'{nm[0]:.3f}+/-{nm[1]:.3f}' if nm else '--'
        print(f'{mg:7.3f} {nms:>16} {f("a_c"):>16} {f("a_c_w"):>17} '
              f'{f("a_c_str"):>14} {f("a_c_wk"):>14}')

    print('\n=== Collapse test input: n(mu_g) vs dZ_eff(mu_g) ===')
    print(f'{"mu_g":>7} {"Z":>7} {"chi":>7} {"Z_c":>7} {"dZ_eff":>8} '
          f'{"n_macro":>9} {"cv_f":>7} {"pr_f":>7}')
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        g = lambda k: float(np.nanmean([r[k] for r in sub]))
        nm = nmac.get(mg)
        print(f'{mg:7.3f} {g("Z"):7.3f} {g("chi"):7.3f} {g("Z_c"):7.3f} '
              f'{g("dZ_eff"):+8.3f} '
              f'{(f"{nm[0]:.3f}" if nm else "--"):>9} '
              f'{g("cv_f"):7.3f} {g("pr_f"):7.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
