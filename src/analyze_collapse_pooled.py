"""
The decisive test of generalised isostaticity: pool TWO independent levers.

Every dZ_eff ever measured in this project was reached by turning one knob,
mu_g. So "n depends on dZ_eff" was never separated from "n and dZ_eff both
depend on mu_g" -- a correlation between two things driven by a common cause
looks exactly like a collapse. On the friction lever alone the collapse sits
at a respectable 1.5 sigma.

Grain stiffness is an orthogonal knob: Emod moves Z at FIXED mu_g and fixed
pressure (Z falls from 4.28 to 3.66 over 1e4..1e6 at mu_g=0.1). It also moves
chi, but only weakly -- chi roughly doubles while Z_c = 3 + chi shifts by only
~0.06, against a Z shift of ~0.63 -- so Z_c cannot absorb the difference and
dZ_eff moves almost as much as Z does.

If n is genuinely a function of dZ_eff, points reached by stiffening grains
must land on the same curve as points reached by adding friction. This script
interpolates the friction-lever curve and measures where the stiffness-lever
points fall relative to it.

GATING. Setpoints are accepted per-SETPOINT on the mean inertial number, not
per-run: a run-level cut against a global median lets individual runs from a
deviant setpoint slip through the tolerance and can flip a fitted n negative.
The coldest setpoint fails this cut in every sweep (I runs 6-35% high), which
is why it is absent throughout.

Usage:  python3 analyze_collapse_pooled.py
"""
import csv
import glob
import os
import re
import numpy as np

I_TOL = 0.03


def load(f):
    rs = list(csv.DictReader(open(f)))
    for r in rs:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return rs


def fit(th, mu):
    th, mu = np.asarray(th, float), np.asarray(mu, float)
    m = (th > 0) & (mu > 0) & np.isfinite(th) & np.isfinite(mu)
    if m.sum() < 3:
        return np.nan, np.nan
    x, y = np.log(th[m]), np.log(mu[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    s2 = np.sum(r ** 2) / max(m.sum() - 2, 1)
    return -c[0], np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0])


def gate_and_fit(runs):
    """Per-setpoint I gate, then fit n over the surviving setpoints."""
    byT = {}
    for r in runs:
        byT.setdefault(r['Tgran'], []).append(r)
    Ts = sorted(byT)
    Im = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
    med = np.median(list(Im.values()))
    keep = [t for t in Ts if abs(Im[t] / med - 1) <= I_TOL]
    if len(keep) < 3:
        return np.nan, np.nan
    sel = [r for t in keep for r in byT[t]]
    return fit([r['Theta'] for r in sel], [r['mu'] for r in sel])


def parse_log(p):
    L = open(p).readlines()
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
    if not rows:
        return None
    a = np.array(rows)[len(rows) // 3:]
    c = {k: a[:, j] for j, k in enumerate(h)}
    return dict(mu=c['v_muI'].mean(), Theta=c['v_Theta'].mean(),
                I=c['v_Iiner'].mean())


def main():
    # friction lever (Emod = 1e5)
    mac, t1 = load('results_matched2d.csv'), load('tier1_matched2d_results.csv')
    FR = []
    for mg in sorted({r['mu_g'] for r in mac}):
        tt = [r for r in t1 if r['mu_g'] == mg]
        if not tt:
            continue
        n, e = gate_and_fit([r for r in mac if r['mu_g'] == mg])
        if not np.isfinite(n):
            continue
        FR.append((float(np.mean([r['dZ_eff'] for r in tt])), n, e, mg))

    # stiffness lever
    st = load('tier1_stiff_results.csv')
    for r in st:
        r['E'] = float(re.match(r'E([0-9.eE+-]+)_mu', r['label']).group(1))
    ST = []
    for E in sorted({r['E'] for r in st}):
        for mg in sorted({r['mu_g'] for r in st}):
            dzr = [r for r in st if r['mu_g'] == mg and r['E'] == E]
            if not dzr:
                continue
            runs = []
            for p in glob.glob(os.path.join('sweep_stiffness', 'log.E*')):
                m = re.match(r'log\.E([0-9.eE+-]+)_mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$',
                             os.path.basename(p))
                if not m or float(m.group(1)) != E or float(m.group(2)) != mg:
                    continue
                q = parse_log(p)
                if q:
                    q['Tgran'] = float(m.group(3))
                    runs.append(q)
            if len({r['Tgran'] for r in runs}) < 3:
                continue
            n, e = gate_and_fit(runs)
            if not np.isfinite(n):
                continue
            ST.append((float(np.mean([r['dZ_eff'] for r in dzr])), n, e, mg, E))

    print('POOLED: is n a single-valued function of dZ_eff?\n')
    print(f'{"dZ_eff":>9}{"n":>9}{"+/-":>7}  lever      state')
    for d, n, e, mg in sorted(FR):
        print(f'{d:>+9.3f}{n:>9.3f}{e:>7.3f}  friction   mu_g={mg:g}')
    for d, n, e, mg, E in sorted(ST):
        print(f'{d:>+9.3f}{n:>9.3f}{e:>7.3f}  stiffness  mu_g={mg:g}, E={E:.0e}')

    fx = np.array([a[0] for a in sorted(FR)])
    fy = np.array([a[1] for a in sorted(FR)])
    print('\nstiffness-lever points vs the friction-lever curve at matched dZ_eff:')
    print('(E=1e5 rows are the friction points themselves -- they agree by '
          'construction and are excluded)')
    sig = []
    for d, n, e, mg, E in sorted(ST):
        if abs(E - 1.0e5) < 1 or d < fx.min() or d > fx.max():
            continue
        pred = float(np.interp(d, fx, fy))
        g = n - pred
        sig.append(g / e)
        print(f'  dZ_eff={d:+.3f}  n={n:.3f}  curve={pred:.3f}  '
              f'gap={g:+.3f} ({abs(g/e):4.1f} sigma)   mu_g={mg:g}, E={E:.0e}')
    if sig:
        rms = float(np.sqrt(np.mean(np.array(sig) ** 2)))
        print(f'\n  RMS discrepancy {rms:.1f} sigma over {len(sig)} independent points')
        print('  VERDICT: ' + ('collapse SURVIVES the orthogonal lever'
                               if rms < 2 else
                               'collapse FAILS -- n is not a function of dZ_eff'))


if __name__ == '__main__':
    main()
