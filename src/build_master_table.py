"""
Assemble every n measurement in the project into one table with its
predictors, so functional forms for n can be tested against ALL of it at once
rather than one sweep at a time.

Each row is a fitted exponent n with the state it was measured in:
    D          dimension (2 or 3)
    mu_g       grain friction coefficient
    Pconf      confining pressure
    kappa      Emod / Pconf, the dimensionless grain stiffness
    kt_kn      tangential / normal stiffness ratio
    chi, Z     microstructure, where a tracking sweep exists (else blank)
    n, n_err   the measurement

Fits use the same per-SETPOINT inertial-number gate used everywhere else: a
setpoint whose mean I departs from the cell median by more than 3% violates
the fixed-I premise the method rests on and is dropped. (Gating per-run
against a global median instead lets individual runs from a deviant setpoint
survive and can flip a fitted n negative.)

Sources, and the reason each is kept separate:
  matched2d    D=2, Pconf=10, friction scan          -- the reference state
  3d           D=3, Pconf=10, friction scan          -- dimension contrast
  stiffness    D=2, Pconf=10, Emod scan              -- moves k_n and k_t together
  kt_P10       D=2, Pconf=10, k_t scan               -- moves k_t alone
  ktgrid_P2    D=2, Pconf=2,  k_t scan               -- older, DIFFERENT state
  steadystate  D=2, Pconf=10, strain 1.5             -- transient control

Pconf is carried as a column rather than merged away because pressure has
repeatedly changed conclusions in this project (it produced the spurious
"framework swap" and inflated the force-weighted fabric result), so rows at
Pconf=2 must never be silently pooled with rows at Pconf=10.

Usage:  python3 build_master_table.py [--out master_n.csv]
"""
import argparse
import csv
import glob
import os
import re
import numpy as np

I_TOL = 0.03
KN_PREFAC = 7.326e4          # 4/3 * E_eff at Emod=1e5, nu=0.3
KT_DEFAULT = 9.05e4          # mindlin NULL at Emod=1e5, nu=0.3
KT_KN_DEFAULT = KT_DEFAULT / KN_PREFAC


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
    """Per-setpoint gates, then fit.

    Two independent ways a cell is unusable, both observed:
      * I drifts from the cell median -- violates the fixed-I premise;
      * the thermostat loses control (Theta exceeds its own setpoint, or
        Theta stops increasing with Tgran) -- then the fit's x-axis is not
        the variable we think it is. This happens at low k_t and high mu_g,
        where a soft Coulomb-limited tangential spring stores large elastic
        energy before slipping; Theta/Tgran reached 5000 in the P=2 k_t grid.
        Without this gate those cells enter a model fit as n = -0.34 or +0.41
        and dominate it.
    """
    byT = {}
    for r in runs:
        byT.setdefault(r['Tgran'], []).append(r)
    Ts = sorted(byT)
    if len(Ts) < 3:
        return np.nan, np.nan
    Im = {t: np.mean([r['I'] for r in byT[t]]) for t in Ts}
    Th = {t: np.mean([r['Theta'] for r in byT[t]]) for t in Ts}
    if max(Th[t] / t for t in Ts if t > 0) > 1.0:
        return np.nan, np.nan
    means = [Th[t] for t in Ts]
    if any(b <= a for a, b in zip(means, means[1:])):
        return np.nan, np.nan
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
    if len(rows) < 4:
        return None
    a = np.array(rows)[len(rows) // 3:]
    c = {k: a[:, j] for j, k in enumerate(h)}
    return dict(mu=c['v_muI'].mean(), Theta=c['v_Theta'].mean(),
                I=c['v_Iiner'].mean())


def from_logs(pattern, rx, keys):
    """Group logs by everything except Tgran/seed, then fit each group."""
    groups = {}
    for p in glob.glob(pattern):
        m = re.match(rx, os.path.basename(p))
        if not m:
            continue
        q = parse_log(p)
        if not q:
            continue
        q['Tgran'] = float(m.group('T'))
        groups.setdefault(tuple(float(m.group(k)) for k in keys), []).append(q)
    return groups


def micro_lookup(path, extra=None):
    """{mu_g: (chi, Z)} or {(extra, mu_g): (chi, Z)} from a tier1 csv."""
    if not os.path.exists(path):
        return {}
    rs = load(path)
    out = {}
    if extra:
        for r in rs:
            r[extra] = float(re.match(r'E([0-9.eE+-]+)_mu', r['label']).group(1))
        for e in {r[extra] for r in rs}:
            for mg in {r['mu_g'] for r in rs}:
                s = [r for r in rs if r[extra] == e and r['mu_g'] == mg]
                if s:
                    out[(e, mg)] = (np.mean([r['chi'] for r in s]),
                                    np.mean([r['Z'] for r in s]))
    else:
        for mg in {r['mu_g'] for r in rs}:
            s = [r for r in rs if r['mu_g'] == mg]
            out[mg] = (np.mean([r['chi'] for r in s]), np.mean([r['Z'] for r in s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='master_n.csv')
    args = ap.parse_args()
    rows = []

    def add(**kw):
        if np.isfinite(kw.get('n', np.nan)):
            rows.append(kw)

    # friction scans (Emod=1e5, default k_t)
    for csvf, micf, D in [('results_matched2d.csv', 'tier1_matched2d_results.csv', 2),
                          ('results_3d.csv', 'tier1_3d_results.csv', 3)]:
        mac = load(csvf)
        mic = micro_lookup(micf)
        for mg in sorted({r['mu_g'] for r in mac}):
            n, e = gate_and_fit([r for r in mac if r['mu_g'] == mg])
            chi, Z = mic.get(mg, (np.nan, np.nan))
            add(source=os.path.basename(csvf).replace('.csv', ''), D=D, mu_g=mg,
                Pconf=10.0, kappa=1.0e5 / 10.0, kt_kn=KT_KN_DEFAULT,
                chi=chi, Z=Z, n=n, n_err=e)

    # Emod scan (k_t/k_n pinned)
    mic = micro_lookup('tier1_stiff_results.csv', extra='E')
    g = from_logs('sweep_stiffness/log.E*',
                  r'log\.E(?P<E>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
                  r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$', ['E', 'mu'])
    for (E, mg), runs in sorted(g.items()):
        n, e = gate_and_fit(runs)
        chi, Z = mic.get((E, mg), (np.nan, np.nan))
        add(source='stiffness', D=2, mu_g=mg, Pconf=10.0, kappa=E / 10.0,
            kt_kn=KT_KN_DEFAULT, chi=chi, Z=Z, n=n, n_err=e)

    # k_t scans (k_n fixed)
    for d, P, tag in [('sweep_kt_P10', 10.0, 'kt_P10'), ('sweep_sens', 2.0, 'ktgrid_P2')]:
        rx = (r'log\.k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
              r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$' if P == 10 else
              r'log\.e0\.5_k(?P<kt>[0-9.eE+-]+)_mu(?P<mu>[0-9.]+)'
              r'_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$')
        for (kt, mg), runs in sorted(from_logs(f'{d}/log.*', rx, ['kt', 'mu']).items()):
            n, e = gate_and_fit(runs)
            add(source=tag, D=2, mu_g=mg, Pconf=P, kappa=1.0e5 / P,
                kt_kn=kt / KN_PREFAC, chi=np.nan, Z=np.nan, n=n, n_err=e)

    # steady-state control
    for (mg,), runs in sorted(from_logs(
            'sweep_steadystate/log.mu*',
            r'log\.mu(?P<mu>[0-9.]+)_T(?P<T>[0-9.eE+-]+)_s(?P<s>\d+)$',
            ['mu']).items()):
        n, e = gate_and_fit(runs)
        add(source='steadystate', D=2, mu_g=mg, Pconf=10.0, kappa=1.0e4,
            kt_kn=KT_KN_DEFAULT, chi=np.nan, Z=np.nan, n=n, n_err=e)

    flds = ['source', 'D', 'mu_g', 'Pconf', 'kappa', 'kt_kn', 'chi', 'Z', 'n', 'n_err']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=flds, restval='')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if isinstance(r.get(k), float)
                            and not np.isfinite(r.get(k)) else r.get(k))
                        for k in flds})
    print(f'wrote {args.out}: {len(rows)} fitted exponents')
    for s in sorted({r['source'] for r in rows}):
        sub = [r for r in rows if r['source'] == s]
        print(f'   {s:<16} {len(sub):>3} rows   '
              f'n range {min(r["n"] for r in sub):+.3f} .. {max(r["n"] for r in sub):+.3f}')


if __name__ == '__main__':
    main()
