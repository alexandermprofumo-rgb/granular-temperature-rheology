"""
Analysis for the 3D frictionless test: same approach as analyze_sweep.py,
extended to a full 3x3 fabric tensor and its deviatoric anisotropy invariant.

Usage:
    python3 analyze_sweep_3d.py --glob "log.*" --out results_3d.csv
"""
import argparse
import glob
import re
import numpy as np
import csv


def parse_log(path, skip_frac=0.3):
    with open(path) as f:
        lines = f.readlines()
    start = None
    for i, l in enumerate(lines):
        if 'BEGIN MEASUREMENT WINDOW' in l:
            start = i
            break
    if start is None:
        return None
    header = None
    rows = []
    for l in lines[start:]:
        toks = l.split()
        if len(toks) >= 6 and toks[0] == 'Step':
            header = toks
            continue
        if header is not None:
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                if rows:
                    break
                continue
            if len(vals) == len(header):
                rows.append(vals)
    if not rows:
        return None
    rows = np.array(rows)
    n_skip = int(len(rows) * skip_frac)
    rows = rows[n_skip:]
    if len(rows) == 0:
        return None
    cols = {h: rows[:, i] for i, h in enumerate(header)}
    return dict(P=np.mean(cols['v_P']), tau=np.mean(cols['v_tau']),
                mu=np.mean(cols['v_muI']), Theta=np.mean(cols['v_Theta']),
                I=np.mean(cols['v_Iiner']), n_samples=len(rows))


def parse_contacts_3d(path, skip_frac=0.3):
    """Full 3x3 fabric tensor F_ij = <n_i n_j> from dumped (dx,dy,dz,fx,fy,fz),
    keeping only rows with nonzero contact force. Anisotropy invariant:
    a = sqrt(3/2 * sum(Fdev_ij^2)) where Fdev = F - I/3 (standard fabric
    anisotropy scalar used in granular mechanics)."""
    frames = []
    cur = []
    with open(path) as f:
        for line in f:
            if line.startswith('ITEM: TIMESTEP'):
                if cur:
                    frames.append(np.array(cur))
                cur = []
                continue
            if line.startswith('ITEM:'):
                continue
            toks = line.split()
            if len(toks) == 7:
                try:
                    cur.append([float(t) for t in toks])
                except ValueError:
                    pass
        if cur:
            frames.append(np.array(cur))

    if not frames:
        return None
    n_skip = int(len(frames) * skip_frac)
    frames = frames[n_skip:]
    if not frames:
        return None

    F = np.zeros((3, 3))
    ncontacts_tot = 0
    for fr in frames:
        if fr.size == 0:
            continue
        dx, dy, dz = fr[:, 1], fr[:, 2], fr[:, 3]
        fx, fy, fz = fr[:, 4], fr[:, 5], fr[:, 6]
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)
        active = fmag > 1e-8
        if not np.any(active):
            continue
        dxa, dya, dza = dx[active], dy[active], dz[active]
        dist = np.sqrt(dxa**2 + dya**2 + dza**2)
        n = np.stack([dxa/dist, dya/dist, dza/dist], axis=1)
        F += n.T @ n
        ncontacts_tot += len(n)

    if ncontacts_tot == 0:
        return None
    F /= ncontacts_tot
    Fdev = F - np.eye(3) / 3.0
    a_c = np.sqrt(1.5 * np.sum(Fdev**2))
    return dict(a_c=a_c, n_contacts=ncontacts_tot / max(len(frames), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='log.*')
    ap.add_argument('--out', default='results_3d.csv')
    ap.add_argument('--label-re', default=r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)')
    args = ap.parse_args()

    pattern = re.compile(args.label_re)
    rows = []
    for logpath in sorted(glob.glob(args.glob)):
        label = logpath.split('.', 1)[1]
        m = pattern.search(label)
        if not m:
            print(f"skip (label didn't match pattern): {logpath}")
            continue
        mu_g, Tgran, seed = m.group(1), m.group(2), m.group(3)
        therm = parse_log(logpath)
        if therm is None:
            print(f"skip (no measurement data): {logpath}")
            continue
        # Contact dumps are bulky and get pruned once a_c has been extracted,
        # so a missing dump is normal for older runs and must not abort the
        # sweep -- the thermodynamic columns (mu, Theta, I) are what n needs.
        try:
            fab = parse_contacts_3d(f"dump.contacts.{label}")
        except FileNotFoundError:
            fab = None
        row = dict(label=label, mu_g=float(mu_g), Tgran=float(Tgran),
                    seed=int(seed), **therm)
        if fab:
            row.update(fab)
        rows.append(row)
        print(f"{label}: mu={therm['mu']:.4f} Theta={therm['Theta']:.4e} "
              f"I={therm['I']:.4e}" + (f" a_c={fab['a_c']:.4f}" if fab else ""))

    if not rows:
        print("No parseable runs found.")
        return

    with open(args.out, 'w', newline='') as f:
        # Union over all rows, not just the first: runs whose contact dump has
        # been pruned carry no a_c/n_contacts, so keying off row 0 drops those
        # columns entirely and then throws on the first row that has them.
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    print("\n--- n(mu_g) fit: n = -d ln(mu)/d ln(Theta) at fixed I ---")
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            print(f"mu_g={mg:.3f}: need >=3 points, have {len(sub)}")
            continue
        x = np.log([r['Theta'] for r in sub])
        y = np.log([r['mu'] for r in sub])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(len(sub) - 2, 1)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(cov[0, 0])
        print(f"mu_g={mg:6.3f}  n={-coef[0]:.4f} +/- {stderr:.4f}  "
              f"<I>={np.mean([r['I'] for r in sub]):.4e}")

    print("\nCompare to: n=1/6=0.1667 (3D, observed/mean-field w/ friction)")


if __name__ == '__main__':
    main()
