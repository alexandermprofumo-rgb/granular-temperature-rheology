"""Is  mu = (a_c+a_n+a_t)/2  a physical identity here, or algebra?

Claim under test: with the code's estimators, the RB sum is EXACTLY the
projection of the stress deviator onto its own axis, divided by P -- i.e.
equal to mu by construction, up to (i) branch-length weighting and (ii)
angular binning.  If so, the quoted 0.9847 +/- 0.0072 measures polydispersity,
not the validity of Rothenburg-Bathurst.
"""
import glob
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_rb import frames

K = 36


def variants(path):
    out = []
    for a in frames(path):
        live = np.linalg.norm(a[:, 6:8], axis=1) > 0
        if live.sum() < 100:
            continue
        b = a[live]
        d = b[:, 4:6]
        L = np.linalg.norm(d, axis=1)
        dn = d / L[:, None]
        fnv, ftv = b[:, 6:8], b[:, 8:10]

        # exactly what analyze_rb.rb_of does
        S = np.einsum('ci,cj->ij', fnv + ftv, d)
        S = 0.5 * (S + S.T)
        P = (S[0, 0] + S[1, 1]) / 2.0
        if P <= 0:
            continue
        dev = complex((S[0, 0] - S[1, 1]) / 2.0, S[0, 1])
        ths = 0.5 * np.angle(dev)
        mu = abs(dev) / P

        th = np.arctan2(dn[:, 1], dn[:, 0]) % np.pi
        that = np.stack([-dn[:, 1], dn[:, 0]], axis=1)
        fs = np.einsum('ij,ij->i', fnv, dn)
        ts = np.einsum('ij,ij->i', ftv, that)
        idx = np.floor(th / np.pi * K).astype(int) % K
        cnt = np.bincount(idx, minlength=K).astype(float)
        if (cnt == 0).any():
            continue
        E = cnt / cnt.sum()
        fnb = np.array([fs[idx == k].mean() for k in range(K)])
        ftb = np.array([ts[idx == k].mean() for k in range(K)])
        tc = (np.arange(K) + 0.5) * np.pi / K
        f0 = np.average(fnb, weights=cnt)
        ph = np.exp(-2j * ths)
        ac = (2 * np.sum(E * np.exp(2j * tc)) * ph).real
        acn = (2 * np.sum(E * fnb * np.exp(2j * tc)) / f0 * ph).real
        at = -(2 * np.sum(E * ftb * np.exp(2j * tc)) / f0 * ph).imag
        rb_code = (acn + at) / 2.0          # == (a_c + a_n + a_t)/2

        # same estimator, UNBINNED (exact contact angles)
        Z = np.mean((fs + 1j * ts) * np.exp(2j * th))
        rb_unbinned = (2 * (Z * ph).real / 2.0) / fs.mean()

        # same estimator, unbinned AND branch-length weighted
        Zl = np.mean(L * (fs + 1j * ts) * np.exp(2j * th))
        rb_lw = (Zl * ph).real / np.mean(L * fs)

        # and its MAGNITUDE (project on its own axis)
        rb_lw_mag = abs(Zl) / np.mean(L * fs)

        out.append((mu, rb_code, rb_unbinned, rb_lw, rb_lw_mag))
    return np.array(out) if out else None


paths = sorted(glob.glob(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'sweep_pichi', 'dump.contacts.P*')))
sel = paths[::12]
A = []
for p in sel:
    v = variants(p)
    if v is not None:
        A.append(v.mean(axis=0))
A = np.array(A)
mu, code, unb, lw, lwm = A.T
print(f'{len(A)} runs sampled from sweep_pichi\n')
for nm, x in (('as coded (binned, no L)', code),
              ('unbinned, no L         ', unb),
              ('unbinned, L-weighted   ', lw),
              ('unbinned, L-wtd, |.|   ', lwm)):
    r = x / mu
    print(f'  RB_sum/mu   {nm}:  {r.mean():.6f} +/- {r.std():.6f}'
          f'   (min {r.min():.6f}, max {r.max():.6f})')
