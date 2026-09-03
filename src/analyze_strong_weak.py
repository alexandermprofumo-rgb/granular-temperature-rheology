"""
Strong/weak contact-network split of the reorientation rate  (roadmap S1 /
campaign item 4) -- the highest-priority open test.

Background. Three microscopic hypotheses were falsified in the original
campaign, and all three shared one assumption: they averaged over every
contact with equal weight. But mu is a stress, i.e. force-weighted, and
granular force transmission is bimodal -- a minority "strong network" of
force chains carries most of the deviatoric stress. The hypothesis is that
an equal-weighted average dilutes, or even inverts, a signal that lives
specifically in the strong sub-network.

This script re-measures Hypothesis 3 (per-episode reorientation rate)
SEPARATELY for above- and below-average-load contacts.

Method. Contacts are tracked directly from dump.contacts.* by their atom-ID
pair, which carries geometry AND normal/tangential force in the same record
-- so, unlike the original attempt, there is no cross-compute ID/force
pairing to get wrong.

For each episode (a maximal run of consecutive frames in which a given ID
pair is in contact):
  * the contact normal angle is treated as a HEADLESS director: successive
    increments are wrapped into (-pi/2, pi/2], so an arbitrary flip of the
    i<->j ordering cannot masquerade as a pi rotation;
  * net_angle  = |sum of wrapped increments|   (matches the original
    convention: net reorientation from first frame to last)
  * path_angle = sum of |wrapped increments|   (total arc actually swept;
    reported alongside because it is the less cancellation-prone measure)
  * rate = angle / lifetime, lifetime in simulation time units;
  * the episode is STRONG if its time-averaged normal force exceeds the
    time-averaged instantaneous frame-mean normal force over the frames it
    was alive, else WEAK.

Episodes shorter than --min-frames are dropped (they cannot support a rate).

Usage:
    python3 analyze_strong_weak.py --dir sweep_tracking_forces \
        --out strong_weak_results.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import numpy as np

NCOL = 11          # index id1 id2 dist dx dy fx fy ftx fty ftmag
DT = 0.001         # timestep


def iter_frames(path):
    txt = open(path).read()
    for blk in txt.split('ITEM: TIMESTEP')[1:]:
        try:
            step = int(blk.strip().split('\n', 1)[0])
        except ValueError:
            continue
        m = re.search(r'ITEM: ENTRIES[^\n]*\n', blk)
        if not m:
            continue
        body = blk[m.end():]
        if not body.strip():
            continue
        try:
            arr = np.array(body.split(), dtype=float)
        except ValueError:
            continue
        if arr.size % NCOL:
            arr = arr[:arr.size - (arr.size % NCOL)]
        if arr.size == 0:
            continue
        yield step, arr.reshape(-1, NCOL)


def wrap_pi(d):
    """Wrap a director angle increment into (-pi/2, pi/2]."""
    return (d + np.pi / 2) % np.pi - np.pi / 2


def analyse_run(path, skip_frac=0.2, min_frames=3):
    frames = list(iter_frames(path))
    if len(frames) < min_frames + 2:
        return None
    frames = frames[int(len(frames) * skip_frac):]

    # per-frame: pair -> (theta, fn); plus the frame's mean normal force
    per_frame, steps, frame_mean_fn = [], [], []
    for step, arr in frames:
        dist, dx, dy = arr[:, 3], arr[:, 4], arr[:, 5]
        fn = np.hypot(arr[:, 6], arr[:, 7])
        act = (fn > 1e-12) & (dist > 0)
        if act.sum() < 10:
            continue
        ids = arr[act][:, 1:3].astype(np.int64)
        theta = np.arctan2(dy[act], dx[act])
        f = fn[act]
        d = {}
        for (a, b), th, ff in zip(ids, theta, f):
            d[(a, b) if a < b else (b, a)] = (th, ff)
        per_frame.append(d)
        steps.append(step)
        frame_mean_fn.append(f.mean())
    if len(per_frame) < min_frames + 1:
        return None

    # walk episodes
    open_ep = {}     # pair -> dict(th_prev, net, path, fsum, fbarsum, n)
    done = []

    def close(ep):
        if ep['n'] >= min_frames:
            done.append(ep)

    for i, d in enumerate(per_frame):
        fbar = frame_mean_fn[i]
        for p, (th, ff) in d.items():
            ep = open_ep.get(p)
            if ep is None:
                open_ep[p] = dict(th_prev=th, net=0.0, path=0.0,
                                  fsum=ff, fbarsum=fbar, n=1, i0=i)
            else:
                dth = wrap_pi(th - ep['th_prev'])
                ep['th_prev'] = th
                ep['net'] += dth
                ep['path'] += abs(dth)
                ep['fsum'] += ff
                ep['fbarsum'] += fbar
                ep['n'] += 1
        for p in list(open_ep):
            if p not in d:
                close(open_ep.pop(p))
    for p in list(open_ep):
        close(open_ep.pop(p))

    if len(done) < 20:
        return None

    dt_frame = (steps[1] - steps[0]) * DT if len(steps) > 1 else DT
    res = {}
    for tag, sel in (('all', lambda e: True),
                     ('strong', lambda e: e['fsum'] / e['n'] > e['fbarsum'] / e['n']),
                     ('weak', lambda e: e['fsum'] / e['n'] <= e['fbarsum'] / e['n'])):
        sub = [e for e in done if sel(e)]
        if len(sub) < 10:
            res[f'rate_{tag}'] = np.nan
            res[f'rate_path_{tag}'] = np.nan
            res[f'n_ep_{tag}'] = len(sub)
            continue
        life = np.array([(e['n'] - 1) * dt_frame for e in sub])
        net = np.abs([e['net'] for e in sub])
        path = np.array([e['path'] for e in sub])
        good = life > 0
        res[f'rate_{tag}'] = float(np.mean(net[good] / life[good]))
        res[f'rate_path_{tag}'] = float(np.mean(path[good] / life[good]))
        res[f'n_ep_{tag}'] = int(good.sum())
        res[f'life_{tag}'] = float(np.mean(life[good]))
    res['frac_strong'] = res['n_ep_strong'] / max(
        res['n_ep_strong'] + res['n_ep_weak'], 1)
    return res


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


def fit(theta, y, sign=+1.0):
    theta = np.asarray(theta, float); y = np.asarray(y, float)
    m = np.isfinite(theta) & np.isfinite(y) & (theta > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, yy = np.log(theta[m]), np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ c
    s2 = np.sum(resid ** 2) / max(m.sum() - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return sign * c[0], np.sqrt(cov[0, 0])


def macroscopic_n(path, core={0.0005, 0.002, 0.006, 0.015}):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    rows = [r for r in rows if r['Tgran'] in core]
    out = {}
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) >= 3:
            out[mg] = fit([r['Theta'] for r in sub], [r['mu'] for r in sub],
                          sign=-1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='sweep_tracking_forces')
    ap.add_argument('--out', default='strong_weak_results.csv')
    ap.add_argument('--macro', default='results.csv')
    ap.add_argument('--min-frames', type=int, default=3)
    args = ap.parse_args()

    pat = re.compile(r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$')
    rows = []
    files = sorted(glob.glob(os.path.join(args.dir, 'dump.contacts.*')))
    if not files:
        print(f'No per-contact dumps under {args.dir}.\n'
              'This script needs tier-3 data, the raw per-contact dumps\n'
              '(~198 GB), which are not distributed. See "Data, and what you\n'
              'need for what" in the README. Nothing in the figure or\n'
              'constraint pipeline depends on this script.', file=sys.stderr)
        return 1
    for i, path in enumerate(files, 1):
        label = os.path.basename(path).split('dump.contacts.', 1)[1]
        m = pat.search(label)
        if not m:
            continue
        res = analyse_run(path, min_frames=args.min_frames)
        if res is None:
            print(f'  skip: {label}', file=sys.stderr)
            continue
        res.update(label=label, mu_g=float(m.group(1)), Tgran=float(m.group(2)),
                   seed=int(m.group(3)))
        th = parse_theta(os.path.join(args.dir, f'log.{label}'))
        res['Theta'] = th if th else res['Tgran']
        rows.append(res)
        print(f'[{i}/{len(files)}] {label}: rate_all={res["rate_all"]:.4f} '
              f'strong={res["rate_strong"]:.4f} weak={res["rate_weak"]:.4f} '
              f'(f_strong={res["frac_strong"]:.2f})')

    if not rows:
        print('nothing parsed')
        return 1
    fields = sorted({k for r in rows for k in r})
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {len(rows)} rows to {args.out}')

    nmac = macroscopic_n(args.macro)
    print('\n=== Theta-scaling of the reorientation rate, split by contact load ===')
    print('(macroscopic n RISES then falls; Hypothesis 3 unsplit FELL monotonically)')
    print(f'{"mu_g":>7} {"n_macro":>16} {"ALL":>16} {"STRONG":>16} {"WEAK":>16}')
    for mg in sorted(set(r['mu_g'] for r in rows)):
        sub = [r for r in rows if r['mu_g'] == mg]
        th = [r['Theta'] for r in sub]
        def f(k):
            v, e = fit(th, [r[k] for r in sub])
            return f'{v:+.3f}+/-{e:.3f}' if np.isfinite(v) else '--'
        nm = nmac.get(mg)
        nms = f'{nm[0]:.3f}+/-{nm[1]:.3f}' if nm else '--'
        print(f'{mg:7.3f} {nms:>16} {f("rate_all"):>16} '
              f'{f("rate_strong"):>16} {f("rate_weak"):>16}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
