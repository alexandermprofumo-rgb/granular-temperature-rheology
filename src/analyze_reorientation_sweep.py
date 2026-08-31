"""
Reuses dump.atoms.* from run_vt_sweep.sh to measure the NET REORIENTATION
ACCUMULATED PER CONTACT EPISODE (birth-to-death angle change of the
contact normal) -- the quantity most directly tied to fabric-mode erasure,
since it's neither pure sliding velocity nor pure persistence time alone,
but their (possibly correlated) combination.

Usage (from inside sweep_vt/):
    python3 analyze_reorientation_sweep.py --glob "dump.atoms.*" --dump-every 15 --out reorientation_results.csv
"""
import argparse
import glob
import re
import csv
import numpy as np
from scipy.spatial import cKDTree


def parse_dump(dumpfile):
    frames = []
    cur_ids, cur_pos, cur_r = [], [], []
    box = None
    with open(dumpfile) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('ITEM: TIMESTEP'):
            if cur_ids:
                frames.append((np.array(cur_ids), np.array(cur_pos), np.array(cur_r), box))
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


def find_contacts_with_normal(ids, pos, r, box):
    Lx, Ly, xy = box
    tree = cKDTree(pos % [Lx, Ly], boxsize=[Lx, Ly])
    rmax = r.max()
    pairs = tree.query_pairs(r=2*rmax, output_type='ndarray')
    out = {}
    if len(pairs) == 0:
        return out
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
        key = (min(ids[ii], ids[jj]), max(ids[ii], ids[jj]))
        out[key] = (dx[k], dy[k])
    return out


def mean_reorientation(frames, dump_every, dt, skip_frac=0.2):
    n_skip = int(len(frames) * skip_frac)
    frames = frames[n_skip:]
    frame_dt = dump_every * dt

    contacts_per_frame = [find_contacts_with_normal(ids, pos, r, box)
                          for ids, pos, r, box in frames]

    active_since = {}   # key -> (start_frame_idx, dx0, dy0)
    episode_reorient = []
    episode_lifetime = []
    all_keys_prev = set()
    for i, contacts in enumerate(contacts_per_frame):
        keys_now = set(contacts.keys())
        for k in keys_now - all_keys_prev:
            dx, dy = contacts[k]
            active_since[k] = (i, dx, dy)
        for k in all_keys_prev - keys_now:
            if k in active_since:
                start_i, dx0, dy0 = active_since.pop(k)
                end_i = i - 1
                dx1, dy1 = contacts_per_frame[end_i][k]
                lifetime = (end_i - start_i + 1) * frame_dt
                th0, th1 = np.arctan2(dy0, dx0), np.arctan2(dy1, dx1)
                dtheta = abs(np.arctan2(np.sin(th1-th0), np.cos(th1-th0)))
                episode_reorient.append(dtheta)
                episode_lifetime.append(lifetime)
        all_keys_prev = keys_now

    if not episode_reorient:
        return None
    episode_reorient = np.array(episode_reorient)
    episode_lifetime = np.array(episode_lifetime)
    # net reorientation rate: total angle change divided by how long it took
    # (avoids div-by-zero: only for episodes lasting >0)
    valid = episode_lifetime > 0
    rate = episode_reorient[valid] / episode_lifetime[valid]
    return dict(mean_reorient=np.mean(episode_reorient),
                mean_rate=np.mean(rate),
                mean_lifetime=np.mean(episode_lifetime),
                n_episodes=len(episode_reorient))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='dump.atoms.*')
    ap.add_argument('--out', default='reorientation_results.csv')
    ap.add_argument('--dt', type=float, default=0.001)
    ap.add_argument('--dump-every', type=int, default=15)
    ap.add_argument('--label-re', default=r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)')
    args = ap.parse_args()

    pattern = re.compile(args.label_re)
    rows = []
    for dumpfile in sorted(glob.glob(args.glob)):
        label = dumpfile.split('dump.atoms.', 1)[1]
        m = pattern.search(label)
        if not m:
            continue
        mu_g, Tgran, seed = m.group(1), m.group(2), m.group(3)
        frames = parse_dump(dumpfile)
        if len(frames) < 3:
            continue
        result = mean_reorientation(frames, args.dump_every, args.dt)
        if result is None:
            continue

        theta = None
        try:
            with open(f'log.{label}') as f:
                lines = f.readlines()
            start = next((i for i, l in enumerate(lines)
                          if 'BEGIN CONTACT TRACKING' in l), None)
            if start is not None:
                vals, header_seen = [], False
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

        row = dict(label=label, mu_g=float(mu_g), Tgran=float(Tgran),
                   Theta=theta if theta else float(Tgran), seed=int(seed), **result)
        rows.append(row)
        print(f"{label}: Theta={theta}  mean_rate={result['mean_rate']:.5f}  "
              f"mean_reorient={result['mean_reorient']:.5f}  n_ep={result['n_episodes']}")

    if not rows:
        print("No parseable runs found.")
        return

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    print("\n--- Theta-scaling exponent of mean_rate (reorientation/lifetime) per mu_g ---")
    print("--- this is the most direct proxy for the erasure rate Gamma itself ---")
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            continue
        x = np.log([r['Theta'] for r in sub])
        y = np.log([r['mean_rate'] for r in sub])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        dof = max(len(sub) - 2, 1)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(cov[0, 0])
        print(f"mu_g={mg:6.3f}  rate_exponent={coef[0]:.4f} +/- {stderr:.4f}")


if __name__ == '__main__':
    main()
