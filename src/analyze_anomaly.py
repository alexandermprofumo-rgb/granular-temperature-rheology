"""
The mu_g ~ 0.3 anomaly: does the pressure-sensitivity spike TRACK the peak of
n(mu_g), or sit at a fixed friction?

kappa = E/P organises n at mu_g = 0.1 and 1.0 but fails at 0.3 (levers
disagree 7x, 4.9 sigma). Two readings survive the evidence so far:

  PEAK PROXIMITY   n is flat at its maximum, so anything shifting the peak
                   makes dn/dln(kappa) look large nearby. Prediction: the
                   spike in b(mu_g) MOVES with the peak as pressure moves it.

  REAL CROSSOVER   something changes character near mu_g ~ 0.3 -- a
                   sticking/sliding transition, say. Prediction: the spike
                   stays PINNED near 0.3 while the peak moves independently.

Both are measured from one dense n(mu_g, P) grid:
  b(mu_g)   = d n / d ln(kappa), fitted across P at each friction
  mu_g*(P)  = peak location, from a parabola through the top three points

The discriminator is the correlation between where the peak sits and where
b peaks, plus whether mu_g* moves at all across the pressure range.

Gating is the same everywhere else in this project: per-SETPOINT cuts on
inertial-number drift, thermostat control, and barostat stability. Gating per
run against a global median instead lets members of a deviant setpoint
survive and can flip a fitted n negative.

Usage:  python3 analyze_anomaly.py
"""
import glob
import os
import re
import numpy as np

I_TOL = 0.05
P_STD_MAX = 0.15


def parse(path):
    L = open(path).readlines()
    i = next((k for k, l in enumerate(L) if 'BEGIN MEASUREMENT WINDOW' in l), None)
    if i is None:
        return None
    h, rows = None, []
    for l in L[i:]:
        t = l.split()
        if t and t[0] == 'Step':
            h = t
            continue
        if h is None:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            if rows:
                break
            continue
        if len(v) == len(h):
            rows.append(v)
    if len(rows) < 4:
        return None
    a = np.array(rows)[len(rows) // 3:]
    c = {k: a[:, j] for j, k in enumerate(h)}
    return dict(mu=c['v_muI'].mean(), Theta=c['v_Theta'].mean(),
                I=c['v_Iiner'].mean(), Pm=c['v_P'].mean(), Ps=c['v_P'].std(),
                Z=c['v_Zc'].mean() if 'v_Zc' in c else np.nan)


def gate_and_fit(runs):
    byT = {}
    for r in runs:
        byT.setdefault(round(r['Tgran'], 10), []).append(r)
    Ts = sorted(byT)
    ok = [t for t in Ts
          if np.mean([r['Ps'] for r in byT[t]]) / np.mean([r['Pm'] for r in byT[t]]) < P_STD_MAX
          and np.mean([r['Theta'] for r in byT[t]]) / t < 1.0]
    if len(ok) < 3:
        return np.nan, np.nan
    Im = {t: np.mean([r['I'] for r in byT[t]]) for t in ok}
    med = np.median(list(Im.values()))
    keep = [t for t in ok if abs(Im[t] / med - 1) <= I_TOL]
    if len(keep) < 3:
        return np.nan, np.nan
    sel = [r for t in keep for r in byT[t]]
    th = np.array([r['Theta'] for r in sel])
    mu = np.array([r['mu'] for r in sel])
    m = (th > 0) & (mu > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], float(np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0]))


def peak_of(mus, ns):
    """Parabola through the three points around the maximum."""
    mus, ns = np.asarray(mus, float), np.asarray(ns, float)
    ok = np.isfinite(ns)
    mus, ns = mus[ok], ns[ok]
    if len(mus) < 3:
        return np.nan, np.nan
    i = int(np.argmax(ns))
    i = min(max(i, 1), len(mus) - 2)
    x, y = mus[i - 1:i + 2], ns[i - 1:i + 2]
    d = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    if abs(d) < 1e-15:
        return float(mus[int(np.argmax(ns))]), float(ns.max())
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / d
    b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0])
         + x[0] ** 2 * (y[1] - y[2])) / d
    if a >= 0:
        return float(mus[int(np.argmax(ns))]), float(ns.max())
    xs = -b / (2 * a)
    return float(np.clip(xs, mus.min(), mus.max())), float(ns.max())


def main():
    R = {}
    for p in glob.glob('sweep_piscan/log.P*'):
        m = re.match(r'log\.P([0-9.]+)_mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$',
                     os.path.basename(p))
        if not m:
            continue
        q = parse(p)
        if not q:
            continue
        P = float(m.group(1))
        q['Tgran'] = float(m.group(3)) * P / 10.0
        R.setdefault((P, float(m.group(2))), []).append(q)

    Ps = sorted({k[0] for k in R if k[0] >= 5.0})
    mus = sorted({k[1] for k in R})
    G = {}
    for (P, mg), runs in R.items():
        if P < 5.0:
            continue
        n, e = gate_and_fit(runs)
        if np.isfinite(n):
            G[(P, mg)] = (n, e)

    print('n(mu_g, P)   [kappa = 1e5/P; higher P = lower kappa]')
    print(f'{"mu_g":>6}' + ''.join(f'{"P=" + str(int(P)):>17}' for P in Ps))
    for mg in mus:
        line = f'{mg:>6g}'
        for P in Ps:
            if (P, mg) in G:
                n, e = G[(P, mg)]
                line += f'{n:+.3f}+/-{e:.3f}'.rjust(17)
            else:
                line += f'{"-":>17}'
        print(line)

    # peak location per pressure
    print('\npeak of n(mu_g) at each pressure')
    print(f'{"P":>6}{"kappa":>10}{"mu_g*":>9}{"n_peak":>9}')
    peaks = {}
    for P in Ps:
        mm = [mg for mg in mus if (P, mg) in G]
        nn = [G[(P, mg)][0] for mg in mm]
        if len(mm) < 3:
            continue
        ms, npk = peak_of(mm, nn)
        peaks[P] = ms
        print(f'{P:>6g}{1e5/P:>10.0f}{ms:>9.3f}{npk:>9.3f}')

    # pressure sensitivity per friction
    print('\nb(mu_g) = dn/dln(kappa), fitted across pressure')
    print(f'{"mu_g":>6}{"b":>12}{"+/-":>9}{"sigma":>8}   (stiffness lever: +0.0066 +/- 0.0023)')
    bs = {}
    for mg in mus:
        pts = [(1e5 / P,) + G[(P, mg)] for P in Ps if (P, mg) in G]
        if len(pts) < 3:
            continue
        x = np.log(np.array([p[0] for p in pts]) / 1e4)
        y = np.array([p[1] for p in pts])
        w = 1.0 / np.array([p[2] for p in pts])
        A = np.vstack([x, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = (A @ c - y) * w
        s2 = (r @ r) / max(len(x) - 2, 1)
        se = float(np.sqrt((s2 * np.linalg.inv((A * w[:, None]).T @ (A * w[:, None])))[0, 0]))
        bs[mg] = (c[0], se)
        z = (c[0] - 0.0066) / np.hypot(se, 0.0023)
        print(f'{mg:>6g}{c[0]:>+12.4f}{se:>9.4f}{abs(z):>8.1f}   '
              + ('AGREE' if abs(z) < 2 else 'DISAGREE'))

    # the discriminator
    print('\n--- PEAK PROXIMITY vs REAL CROSSOVER ---')
    if peaks:
        pv = np.array([peaks[P] for P in Ps if P in peaks])
        print(f'peak location mu_g* across pressure: '
              f'{", ".join(f"{peaks[P]:.3f}" for P in Ps if P in peaks)}')
        print(f'   spread {pv.max() - pv.min():.3f} '
              f'(grid spacing is {np.diff(sorted(mus)).min():.2f})')
    if bs:
        bmax = max(bs, key=lambda k: bs[k][0])
        print(f'b(mu_g) peaks at mu_g = {bmax:g}  (b = {bs[bmax][0]:+.4f})')
        if peaks:
            mean_peak = float(np.mean([peaks[P] for P in Ps if P in peaks]))
            print(f'mean peak location      = {mean_peak:.3f}')
            print(f'   separation |b-peak - n-peak| = {abs(bmax - mean_peak):.3f}')
        print("""
Reading:
  * mu_g* essentially FIXED across pressure and b spiking AWAY from it
      -> the spike is not peak proximity; a real crossover is implicated.
  * mu_g* MOVING with pressure and b spiking where the peak sits
      -> peak proximity explains it; no new physics needed.
  * mu_g* fixed AND b spiking exactly at it -> degenerate, needs the
      stiffness lever's peak (which does not move n's peak) to break.""")


if __name__ == '__main__':
    main()
