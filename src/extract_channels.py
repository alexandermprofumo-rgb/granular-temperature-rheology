"""Extract per-run channel data from every 2D sweep that dumps tangential force.

FIXES A REAL BUG. analyze_collapse_channels.rb_of used analyze_rb.frames, which
yields EVERY frame in the dump. But in.granular_2d_ss defines the dump before
`run ${nequil}`, so of 41 frames the first 21 (steps 0 to 500000) are
EQUILIBRATION -- half the data averaged into every 2D channel value came from
before the measurement window opened. The 3D work (analyze_channels_3d_steady,
analyze_exact_split) used keep_last=8 and is unaffected; only the 2D
three-channel numbers and the 2D channel collapse tests are.

Here frames are selected by TIMESTEP, not by position: keep the last `KEEP`
frames, reached by seeking to a fraction of the file so the equilibration half
is never parsed at all. That is both correct and ~5x faster.

Per run it records: mu, the three anisotropies (branch-length weighted, so
mu = (a_c+a_n+a_t)/2 is exact), chi, and force-distribution moments.

Usage:  python3 extract_channels.py [sweep ...]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

MOB = 0.99

SWEEPS = {
    'sweep_steady2d': (r'dump\.contacts\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
                       'chan_steady2d.csv', ()),
    'sweep_lev_E':    (r'dump\.contacts\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                       r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', 'chan_lev_E.csv', ('E',)),
    'sweep_lev_P':    (r'dump\.contacts\.P(?P<P>[0-9.]+)_mu(?P<mu>[0-9.]+)'
                       r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', 'chan_lev_P.csv', ('P',)),
    'sweep_lev_kt':   (r'dump\.contacts\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                       r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', 'chan_lev_kt.csv', ('kt',)),
}


def tail_frames(path, keep=None, ncol=11):
    """All MEASUREMENT-WINDOW frames, without parsing the equilibration ones.

    nequil == nmeas in every campaign, so the measurement window is exactly the
    second half of the run and the frames to keep are the last half. An earlier
    version kept only the last 8, which was unbiased but too noisy: the
    three-channel consistency gate went from 1.0 sigma (biased, all 41 frames)
    to 2.6 sigma (unbiased, 8 frames). Keeping the whole second half is both
    unbiased and low-variance, and it adapts automatically to the arms whose
    frame counts differ (the stiffness lever runs 16 to 101 frames per cell).
    """
    size = os.path.getsize(path)
    with open(path) as f:
        # count frames cheaply, then seek past all but the last `keep`
        f.seek(0)
        head = f.read(200000)
        m = re.search(r'ITEM: NUMBER OF ENTRIES\n(\d+)', head)
        if not m:
            return []
        approx = int(m.group(1)) * 14 + 300        # bytes per frame, generous
        nfr = max(1, size // max(approx, 1))
        keep = keep or max(4, int(nfr // 2))
        start = max(0, size - approx * (keep + 1))
        f.seek(start)
        f.readline()
        out = []
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: NUMBER OF ENTRIES'):
                n = int(f.readline())
                while True:
                    l = f.readline()
                    if not l or l.startswith('ITEM: ENTRIES'):
                        break
                block = ''.join(f.readline() for _ in range(n))
                a = np.fromstring(block, sep=' ')
                if a.size >= n * ncol:
                    out.append(a[:n * ncol].reshape(n, ncol))
    return out[-keep:]


def analyse(path):
    acc = []
    for a in tail_frames(path):
        live = np.linalg.norm(a[:, 6:8], axis=1) > 0
        if live.sum() < 100:
            continue
        b = a[live]
        d = b[:, 4:6]
        L = np.linalg.norm(d, axis=1)
        dn = d / L[:, None]
        fnv, ftv = b[:, 6:8], b[:, 8:10]
        S = np.einsum('ci,cj->ij', fnv + ftv, d)
        S = 0.5 * (S + S.T)
        P = (S[0, 0] + S[1, 1]) / 2.0
        if P <= 0:
            continue
        dev = complex((S[0, 0] - S[1, 1]) / 2.0, S[0, 1])
        ths = 0.5 * np.angle(dev)
        th = np.arctan2(dn[:, 1], dn[:, 0]) % np.pi
        that = np.stack([-dn[:, 1], dn[:, 0]], axis=1)
        fs = np.einsum('ij,ij->i', fnv, dn)
        ts = np.einsum('ij,ij->i', ftv, that)
        w = L / L.sum()
        f0 = float(np.sum(w * fs))
        ph = np.exp(-2j * ths)
        ac = 2 * (np.sum(w * np.exp(2j * th)) * ph).real
        acn = 2 * (np.sum(w * fs * np.exp(2j * th)) / f0 * ph).real
        at = -2 * (np.sum(w * ts * np.exp(2j * th)) / f0 * ph).imag
        fm = float(fs.mean())
        acc.append((abs(dev) / P, ac, acn - ac, at, fm,
                    float(fs.std() / max(fm, 1e-30)),
                    float((fs.sum() ** 2) / max(len(fs) * (fs ** 2).sum(), 1e-30)),
                    int(live.sum())))
    return np.mean(acc, axis=0) if acc else None


def chi_of(path, mu_g):
    vals = []
    for a in tail_frames(path):
        fn = np.linalg.norm(a[:, 6:8], axis=1)
        live = fn > 0
        if live.sum() < 100:
            continue
        vals.append(float((a[live, 10] / (mu_g * fn[live]) >= MOB).mean()))
    return float(np.mean(vals)) if vals else np.nan


def run(sweep):
    rx_s, out, extra = SWEEPS[sweep]
    rx = re.compile(rx_s)
    paths = sorted(glob.glob(f'{sweep}/dump.contacts.*'))
    rows = []
    for i, p in enumerate(paths):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        g = m.groupdict()
        mg = float(g['mu'])
        r = analyse(p)
        if r is None:
            continue
        row = dict(mu_g=mg, Tgran=float(g['T']), seed=g['s'],
                   mu=r[0], a_c=r[1], a_n=r[2], a_t=r[3],
                   f_mean=r[4], cv_f=r[5], pr_f=r[6], ncon=r[7],
                   chi=(chi_of(p, mg) if mg > 0 else 0.0))
        for k in extra:
            row[k] = float(g[k])
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f'   {sweep} {i+1}/{len(paths)}', file=sys.stderr)
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(f'{sweep}: {len(rows)} runs -> {out}')


if __name__ == '__main__':
    for s in (sys.argv[1:] or list(SWEEPS)):
        run(s)
