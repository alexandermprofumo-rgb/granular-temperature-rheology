"""
Parse the contact-tracking dumps from run_vt_sweep.sh, extract mean |vt|
(relative tangential sliding velocity) per (mu_g, Tgran, seed), fit the
Theta-scaling exponent of |vt| for each mu_g, and compare against the
macroscopic n(mu_g) curve already measured from mu(Theta) fits.

Usage:
    python3 analyze_vt_sweep.py --glob "dump.atoms.*" --out vt_sweep_results.csv
"""
import argparse
import glob
import re
import csv
import numpy as np
from scipy.spatial import cKDTree


def parse_dump(dumpfile):
    frames = []
    cur_ids, cur_pos, cur_vel, cur_omega, cur_r = [], [], [], [], []
    box = None
    with open(dumpfile) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('ITEM: TIMESTEP'):
            if cur_ids:
                frames.append((np.array(cur_ids), np.array(cur_pos), np.array(cur_vel),
                                np.array(cur_omega), np.array(cur_r), box))
            cur_ids, cur_pos, cur_vel, cur_omega, cur_r = [], [], [], [], []
            i += 2
            continue
        if line.startswith('ITEM: BOX BOUNDS'):
            xline = lines[i+1].split()
            yline = lines[i+2].split()
            xlo, xhi, xy = float(xline[0]), float(xline[1]), float(xline[2])
            ylo, yhi = float(yline[0]), float(yline[1])
            Lx, Ly = xhi - xlo, yhi - ylo
            box = (Lx, Ly, xy)
            i += 4
            continue
        if line.startswith('ITEM: ATOMS'):
            i += 1
            continue
        toks = line.split()
        if len(toks) == 7:
            try:
                aid = int(toks[0])
                x, y, vx, vy, omz, r = [float(t) for t in toks[1:7]]
                cur_ids.append(aid); cur_pos.append((x, y)); cur_vel.append((vx, vy))
                cur_omega.append(omz); cur_r.append(r)
            except ValueError:
                pass
        i += 1
    if cur_ids:
        frames.append((np.array(cur_ids), np.array(cur_pos), np.array(cur_vel),
                        np.array(cur_omega), np.array(cur_r), box))
    return frames


def mean_vt(frames, skip_frac=0.2):
    n_skip = int(len(frames) * skip_frac)
    frames = frames[n_skip:]
    all_vt = []
    for ids, pos, vel, omega, r, box in frames:
        Lx, Ly, xy = box
        tree = cKDTree(pos % [Lx, Ly], boxsize=[Lx, Ly])
        rmax = r.max()
        pairs = tree.query_pairs(r=2*rmax, output_type='ndarray')
        if len(pairs) == 0:
            continue
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        dx = pos[i_idx, 0] - pos[j_idx, 0]
        dy = pos[i_idx, 1] - pos[j_idx, 1]
        n = np.round(dy / Ly)
        dx = dx - n * xy
        dy = dy - n * Ly
        dx = dx - Lx * np.round(dx / Lx)
        dist = np.hypot(dx, dy)
        sum_r = r[i_idx] + r[j_idx]
        overlap = sum_r - dist
        mask = overlap > 0
        if not np.any(mask):
            continue
        ii, jj = i_idx[mask], j_idx[mask]
        dxm, dym, distm = dx[mask], dy[mask], dist[mask]
        nx, ny = dxm/distm, dym/distm
        rvx = vel[ii,0] - vel[jj,0]
        rvy = vel[ii,1] - vel[jj,1]
        vt = (rvx*(-ny) + rvy*nx) + r[ii]*omega[ii] + r[jj]*omega[jj]
        all_vt.extend(np.abs(vt).tolist())
    if not all_vt:
        return None
    return np.mean(all_vt), np.median(all_vt), len(all_vt)


def parse_log_theta(logfile, skip_frac=0.3):
    """Mean measured Theta (c_tbias) during the tracking window."""
    try:
        with open(logfile) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None
    start = None
    for i, l in enumerate(lines):
        if 'BEGIN CONTACT TRACKING' in l:
            start = i
            break
    if start is None:
        return None
    header = None
    rows = []
    for l in lines[start:]:
        toks = l.split()
        if len(toks) == 3 and toks[0] == 'Step':
            header = toks
            continue
        if header is not None:
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                if rows:
                    break
                continue
            if len(vals) == 3:
                rows.append(vals)
    if not rows:
        return None
    rows = np.array(rows)
    n_skip = int(len(rows) * skip_frac)
    rows = rows[n_skip:]
    if len(rows) == 0:
        return None
    return np.mean(rows[:, 2])  # columns: step, v_P, c_tbias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='dump.atoms.*')
    ap.add_argument('--out', default='vt_sweep_results.csv')
    ap.add_argument('--label-re', default=r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)')
    args = ap.parse_args()

    pattern = re.compile(args.label_re)
    rows = []
    for dumpfile in sorted(glob.glob(args.glob)):
        label = dumpfile.split('dump.atoms.', 1)[1]
        m = pattern.search(label)
        if not m:
            print(f"skip (label didn't match): {dumpfile}")
            continue
        mu_g, Tgran, seed = m.group(1), m.group(2), m.group(3)
        frames = parse_dump(dumpfile)
        if len(frames) < 3:
            print(f"skip (too few frames): {dumpfile}")
            continue
        result = mean_vt(frames)
        if result is None:
            continue
        mean_v, med_v, n = result
        theta_measured = parse_log_theta(f'log.{label}')
        rows.append(dict(label=label, mu_g=float(mu_g), Tgran=float(Tgran),
                          Theta=theta_measured if theta_measured else float(Tgran),
                          seed=int(seed), mean_vt=mean_v, median_vt=med_v, n_contacts=n))
        print(f"{label}: Theta={theta_measured}  mean|vt|={mean_v:.5f}  n={n}")

    if not rows:
        print("No parseable runs found.")
        return

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    print("\n--- vt-scaling exponent per mu_g: d ln(mean|vt|) / d ln(Theta) ---")
    mu_list = sorted(set(r['mu_g'] for r in rows))
    for mg in mu_list:
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            print(f"mu_g={mg}: not enough points")
            continue
        x = np.log([r['Theta'] for r in sub])
        y = np.log([r['mean_vt'] for r in sub])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(len(sub) - 2, 1)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(cov[0, 0])
        print(f"mu_g={mg:6.3f}  vt_scaling_exponent={coef[0]:.4f} +/- {stderr:.4f}")


if __name__ == '__main__':
    main()
