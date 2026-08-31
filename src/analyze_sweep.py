"""
Parse the LAMMPS logs + contact dumps produced by run_sweep.sh, and fit
n(mu_g) = -d ln(mu) / d ln(Theta) at fixed I, for direct comparison against
the paper's predictions:
    n = 1/6 in 3D (not tested here, 2D-only setup)
    n = 1/8 in 2D at finite friction (observed, Irmer et al./Kim & Kamrin)
    n -> 1/4 in 2D predicted by the Z2 hypothesis as mu_g -> 0

Usage:
    python3 analyze_sweep.py --glob "log.*" --contacts-glob "dump.contacts.*"
"""
import argparse
import glob
import re
import numpy as np
import csv


def parse_log(path, skip_frac=0.3):
    """Average thermo columns over the measurement window (after
    '=== BEGIN MEASUREMENT WINDOW' marker), discarding the first skip_frac
    of that window as further equilibration transient."""
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
                if toks and toks[0].isdigit() is False:
                    # end of thermo block (loop time, etc.)
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
        rows = np.array(rows)

    cols = {h: rows[:, i] for i, h in enumerate(header)}
    return dict(
        P=np.mean(cols['v_P']),
        tau=np.mean(cols['v_tau']),
        mu=np.mean(cols['v_muI']),
        Theta=np.mean(cols['v_Theta']),
        I=np.mean(cols['v_Iiner']),
        n_samples=len(rows),
    )


def parse_contacts(path, skip_frac=0.3):
    """Compute the fabric anisotropy magnitude a_c from dumped per-contact
    (dx, dy, fx, fy) local data, averaged over frames in the dump file,
    keeping only rows with nonzero contact force (pair/local reports all
    neighbor pairs within the interaction cutoff, not just active
    contacts)."""
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
            if len(toks) == 5:
                try:
                    vals = [float(t) for t in toks]
                    cur.append(vals)
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

    Fxx, Fxy, Fyy, ncontacts_tot = 0.0, 0.0, 0.0, 0
    for fr in frames:
        if fr.size == 0:
            continue
        dx, dy, fx, fy = fr[:, 1], fr[:, 2], fr[:, 3], fr[:, 4]
        fmag = np.sqrt(fx**2 + fy**2)
        active = fmag > 1e-8
        if not np.any(active):
            continue
        dxa, dya = dx[active], dy[active]
        dist = np.sqrt(dxa**2 + dya**2)
        nx, ny = dxa / dist, dya / dist
        Fxx += np.sum(nx * nx)
        Fxy += np.sum(nx * ny)
        Fyy += np.sum(ny * ny)
        ncontacts_tot += len(nx)

    if ncontacts_tot == 0:
        return None
    Fxx /= ncontacts_tot
    Fxy /= ncontacts_tot
    Fyy /= ncontacts_tot
    a_c = 2 * np.sqrt(((Fxx - Fyy) / 2) ** 2 + Fxy ** 2)
    return dict(a_c=a_c, n_contacts=ncontacts_tot / max(len(frames), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='log.*')
    ap.add_argument('--contacts-glob', default='dump.contacts.*')
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--label-re', default=r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)',
                     help='regex to pull mu_g, Tgran, seed out of the label')
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

        contacts_path = f"dump.contacts.{label}"
        fab = parse_contacts(contacts_path)

        row = dict(label=label, mu_g=float(mu_g), Tgran=float(Tgran),
                    seed=int(seed), **therm)
        if fab:
            row.update(fab)
        rows.append(row)
        print(f"{label}: mu={therm['mu']:.4f} Theta={therm['Theta']:.4e} "
              f"I={therm['I']:.4e}" +
              (f" a_c={fab['a_c']:.4f}" if fab else " a_c=NA"))

    if not rows:
        print("No parseable runs found.")
        return

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # fit n = -d ln(mu) / d ln(Theta) at fixed I, per mu_g
    print("\n--- n(mu_g) fit: n = -d ln(mu)/d ln(Theta) at fixed shear rate ---")
    mu_g_values = sorted(set(r['mu_g'] for r in rows))
    print(f"{'mu_g':>8} {'n_fit':>10} {'stderr':>10} {'<I>':>12} {'n_points':>9}")
    for mg in mu_g_values:
        sub = [r for r in rows if r['mu_g'] == mg]
        if len(sub) < 3:
            print(f"{mg:8.4f} {'--':>10} {'--':>10} {'--':>12} {len(sub):9d}"
                  "  (need >=3 Theta points to fit)")
            continue
        lnTheta = np.log(np.array([r['Theta'] for r in sub]))
        lnMu = np.log(np.array([r['mu'] for r in sub]))
        A = np.vstack([lnTheta, np.ones_like(lnTheta)]).T
        coef, res, rank, sv = np.linalg.lstsq(A, lnMu, rcond=None)
        slope = coef[0]
        # rough stderr from residuals
        resid = lnMu - A @ coef
        dof = max(len(sub) - 2, 1)
        s2 = np.sum(resid**2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(cov[0, 0])
        meanI = np.mean([r['I'] for r in sub])
        print(f"{mg:8.4f} {-slope:10.4f} {stderr:10.4f} {meanI:12.4e} {len(sub):9d}")

    print("\nCompare to: n=1/8=0.125 (observed, finite friction), "
          "n=1/4=0.25 (mean-field / frictionless prediction)")


if __name__ == '__main__':
    main()
