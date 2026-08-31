"""
Reuses the dump.atoms.* files already produced by run_vt_sweep.sh (no new
simulations needed) to extract contact PERSISTENCE TIME statistics --
distinct from the vt (sliding velocity) statistics already analyzed -- and
fit its Theta-scaling exponent per mu_g, for comparison against the
macroscopic n(mu_g) curve.

Run this from inside the same directory as run_vt_sweep.sh's sweep_vt/
output (i.e. cd sweep_vt first).

Usage:
    python3 analyze_persistence_sweep.py --glob "dump.atoms.*" --out persistence_results.csv
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
                frames.append((np.array(cur_ids), np.array(cur_pos),
                                np.array(cur_r), box))
            cur_ids, cur_pos, cur_r = [], [], []
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
                x, y = float(toks[1]), float(toks[2])
                r = float(toks[6])
                cur_ids.append(aid); cur_pos.append((x, y)); cur_r.append(r)
            except ValueError:
                pass
        i += 1
    if cur_ids:
        frames.append((np.array(cur_ids), np.array(cur_pos), np.array(cur_r), box))
    return frames


def find_contact_keys(ids, pos, r, box):
    Lx, Ly, xy = box
    tree = cKDTree(pos % [Lx, Ly], boxsize=[Lx, Ly])
    rmax = r.max()
    pairs = tree.query_pairs(r=2*rmax, output_type='ndarray')
    keys = set()
    if len(pairs) == 0:
        return keys
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
    for k in np.where(mask)[0]:
        ii, jj = i_idx[k], j_idx[k]
        keys.add((min(ids[ii], ids[jj]), max(ids[ii], ids[jj])))
    return keys


def mean_persistence(frames, dump_every, dt, skip_frac=0.2):
    n_skip = int(len(frames) * skip_frac)
    frames = frames[n_skip:]
    frame_dt = dump_every * dt

    all_keys_per_frame = [find_contact_keys(ids, pos, r, box)
                           for ids, pos, r, box in frames]

    active_since = {}
    lifetimes = []
    all_keys_prev = set()
    for i, keys_now in enumerate(all_keys_per_frame):
        for k in keys_now - all_keys_prev:
            active_since[k] = i
        for k in all_keys_prev - keys_now:
            if k in active_since:
                start_i = active_since.pop(k)
                lifetime = (i - 1 - start_i + 1) * frame_dt
                lifetimes.append(lifetime)
        all_keys_prev = keys_now

    if not lifetimes:
        return None
    return np.mean(lifetimes), np.median(lifetimes), len(lifetimes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='dump.atoms.*')
    ap.add_argument('--out', default='persistence_results.csv')
    ap.add_argument('--dt', type=float, default=0.001)
    ap.add_argument('--dump-every', type=int, default=15,
                     help='must match the dump_every used in run_vt_sweep.sh')
    ap.add_argument('--label-re', default=r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)')
    args = ap.parse_args()

    pattern = re.compile(args.label_re)
    rows = []
    for dumpfile in sorted(glob.glob(args.glob)):
        label = dumpfile.split('dump.atoms.', 1)[1]
        m = pattern.search(label)
        if not m:
            print(f"skip: {dumpfile}")
            continue
        mu_g, Tgran, seed = m.group(1), m.group(2), m.group(3)
        frames = parse_dump(dumpfile)
        if len(frames) < 3:
            print(f"skip (too few frames): {dumpfile}")
            continue
        result = mean_persistence(frames, args.dump_every, args.dt)
        if result is None:
            continue
        mean_t, med_t, n = result

        # pull actual measured Theta from the corresponding log file
        theta = None
        try:
            with open(f'log.{label}') as f:
                lines = f.readlines()
            start = next((i for i, l in enumerate(lines)
                          if 'BEGIN CONTACT TRACKING' in l), None)
            if start is not None:
                vals = []
                header_seen = False
                for l in lines[start:]:
                    toks = l.split()
                    if len(toks) == 3 and toks[0] == 'Step':
                        header_seen = True
                        continue
                    if header_seen:
                        try:
                            vals.append([float(t) for t in toks])
                        except ValueError:
                            if vals:
                                break
                if vals:
                    vals = np.array(vals)
                    theta = np.mean(vals[int(len(vals)*0.3):, 2])
        except FileNotFoundError:
            pass

        rows.append(dict(label=label, mu_g=float(mu_g), Tgran=float(Tgran),
                          Theta=theta if theta else float(Tgran),
                          seed=int(seed), mean_persist=mean_t,
                          median_persist=med_t, n_episodes=n))
        print(f"{label}: Theta={theta}  mean_persist={mean_t:.5f}  n_episodes={n}")

    if not rows:
        print("No parseable runs found.")
        return

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    print("\n--- persistence-time Theta-scaling exponent per mu_g ---")
    print("(note: this is the exponent of MEAN LIFETIME vs Theta -- if")
    print(" contacts live for SHORTER times at higher Theta, as expected,")
    print(" this exponent is NEGATIVE)")
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            continue
        x = np.log([r['Theta'] for r in sub])
        y = np.log([r['mean_persist'] for r in sub])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(len(sub) - 2, 1)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(cov[0, 0])
        print(f"mu_g={mg:6.3f}  persistence_exponent={coef[0]:.4f} +/- {stderr:.4f}")


if __name__ == '__main__':
    main()
