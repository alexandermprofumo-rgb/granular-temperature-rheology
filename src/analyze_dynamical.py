"""
DYNAMICAL contact observables -- the one class this project has never measured.

Every observable tried so far (Z, chi, dZ_eff, a_c, a_c^w, cv_f, pr_f, f_mean)
is a STATIC snapshot of the contact network, and every one has failed a
two-lever test. But the sharpest signal in the data is dynamical: n rises with
normal stiffness and FALLS with tangential stiffness. Trying to explain a
contact-mechanical signature with network geometry is plausibly why eight
candidates in a row have failed.

Three observables, none of which the project has recorded:

  tau_c   mean contact LIFETIME, in strain units (gdot * duration). How long a
          contact survives relative to the deformation that destroys it.

  s_slip  time-integrated SLIP FRACTION: for each contact episode, the
          fraction of its own lifetime spent at the Coulomb threshold,
          averaged over episodes. This is NOT chi. chi is the fraction of
          contacts sliding at an instant -- a snapshot, and lifetime-weighted.
          s_slip weights every episode equally, so a population of many short
          sliding contacts and one long sticking one gives a large s_slip but
          a small chi. They are different observables and can disagree.

  R_E     ratio of stored TANGENTIAL to NORMAL elastic energy, summed over
          contacts: (f_t^2/k_t) / (f_n^2/k_n). This is the quantity the
          opposite-signed stiffness derivatives most directly implicate, since
          it is the only measured thing in which k_n and k_t appear with
          opposite powers.

All three come from the EXISTING force-resolved dumps (id1 id2 dist dx dy fx
fy ftx fty ftmag every cdump_every steps), so a first look needs no new
simulation. Contacts are tracked by sorted (id1,id2) across frames; an episode
ends when the pair is absent for a frame.

CONTACT LIFETIME IS NOT MEASURABLE FROM THIS DATA -- do not try. The tracking
runs are ntrack=3000 steps at dt=0.001 and gdot=1e-3, i.e. a total strain of
0.003. Contacts are destroyed by deformation of order strain ~ 0.1, so
essentially none break inside the window: tau comes out at 79-80 frames of an
80-frame window, at EVERY friction. That is the window length, not a
lifetime. Measuring tau_c needs a dedicated run with ntrack ~ 1e5 steps
(strain ~0.1); tau_frames is kept below only as a saturation diagnostic.

Usage:
  python3 analyze_dynamical.py --dir sweep_tracking_matched2d --out dyn_fric.csv
  python3 analyze_dynamical.py --dir sweep_tracking_stiff     --out dyn_stiff.csv
"""
import argparse
import csv
import glob
import os
import re
import numpy as np

NCOL2D = 11
TOL = 0.01          # within 1% of the Coulomb threshold counts as sliding


def iter_frames(path, ncol=NCOL2D):
    with open(path) as f:
        txt = f.read()
    for blk in txt.split('ITEM: TIMESTEP')[1:]:
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
        if arr.size < ncol:
            continue
        arr = arr[:arr.size - (arr.size % ncol)]
        yield arr.reshape(-1, ncol)


def analyse(path, mu_g, kt_kn, skip_frac=0.2):
    frames = list(iter_frames(path))
    if len(frames) < 5:
        return None
    frames = frames[int(len(frames) * skip_frac):]

    # episode[(i,j)] = [n_frames_alive, n_frames_sliding]
    live, done = {}, []
    prev = set()
    for arr in frames:
        # columns: 0 index, 1 id1, 2 id2, 3 dist, 4 dx, 5 dy, 6 fx, 7 fy,
        # 8 ftx, 9 fty, 10 ftmag.  The NORMAL force is 6:8 -- 4:6 is the
        # separation vector, and using it silently turns the Coulomb
        # threshold into mu_g * |d| ~ mu_g, which is meaningless.
        fn = np.linalg.norm(arr[:, 6:8], axis=1)
        act = (fn > 1e-12) & (arr[:, 3] > 0)
        if act.sum() < 10:
            continue
        ids = arr[act][:, 1:3]
        ftm = arr[act][:, -1]
        fna = fn[act]
        thresh = mu_g * fna
        sliding = (ftm >= (1 - TOL) * thresh) if mu_g > 0 else np.ones(len(fna), bool)
        cur = set()
        for (a, b), sl in zip(ids, sliding):
            k = (int(min(a, b)), int(max(a, b)))
            cur.add(k)
            if k in live:
                live[k][0] += 1
                live[k][1] += int(sl)
            else:
                live[k] = [1, int(sl)]
        for k in prev - cur:
            if k in live:
                done.append(live.pop(k))
        prev = cur
    done.extend(live.values())
    if len(done) < 50:
        return None
    dur = np.array([d[0] for d in done], float)
    slp = np.array([d[1] for d in done], float)

    # energy ratio, pooled over the same frames
    Et, En = 0.0, 0.0
    for arr in frames:
        fn = np.linalg.norm(arr[:, 6:8], axis=1)
        act = (fn > 1e-12) & (arr[:, 3] > 0)
        if act.sum() < 10:
            continue
        Et += float(np.sum(arr[act][:, -1] ** 2))
        En += float(np.sum(fn[act] ** 2))
    # f_t^2/k_t over f_n^2/k_n  ->  (Et/En) * (k_n/k_t)
    R_E = (Et / En) / kt_kn if En > 0 else np.nan

    return dict(tau_frames=float(dur.mean()),
                tau_median=float(np.median(dur)),
                s_slip=float(np.mean(slp / dur)),
                s_slip_lw=float(slp.sum() / dur.sum()),   # lifetime-weighted
                n_episodes=len(done),
                R_E=R_E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kt-kn', type=float, default=1.2353)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, 'dump.contacts.*'))):
        lab = os.path.basename(p).split('dump.contacts.', 1)[1]
        m = re.search(r'mu([0-9.]+)_T([0-9.eE+-]+)_s(\d+)$', lab)
        if not m:
            continue
        mu_g = float(m.group(1))
        mE = re.match(r'E([0-9.eE+-]+)_', lab)
        # k_t may vary within a sweep (sweep_tracking_kt). R_E divides by
        # k_t/k_n, so using one global value there would fold the very lever
        # being tested into the observable and guarantee a spurious pass.
        mK = re.match(r'k([0-9.eE+-]+)_', lab)
        ktkn = float(mK.group(1)) / 7.326e4 if mK else args.kt_kn
        r = analyse(p, mu_g, ktkn)
        if r is None:
            continue
        r.update(label=lab, mu_g=mu_g, Tgran=float(m.group(2)),
                 seed=int(m.group(3)),
                 Emod=float(mE.group(1)) if mE else 1.0e5,
                 kt_kn=ktkn)
        rows.append(r)

    if not rows:
        print(f'no usable dumps in {args.dir}')
        return
    flds = ['label', 'mu_g', 'Tgran', 'seed', 'Emod', 'kt_kn', 'tau_frames',
            'tau_median', 's_slip', 's_slip_lw', 'n_episodes', 'R_E']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out}: {len(rows)} runs')

    keys = sorted({(r['kt_kn'], r['mu_g']) for r in rows})
    print(f'\n{"k_t/k_n":>9}{"mu_g":>7}{"tau(frames)":>13}{"s_slip":>9}'
          f'{"s_slip_lw":>11}{"R_E":>9}')
    for E, mg in keys:
        s = [r for r in rows if r['kt_kn'] == E and r['mu_g'] == mg]
        print(f'{E:>9.3f}{mg:>7g}{np.mean([r["tau_frames"] for r in s]):>13.2f}'
              f'{np.mean([r["s_slip"] for r in s]):>9.4f}'
              f'{np.mean([r["s_slip_lw"] for r in s]):>11.4f}'
              f'{np.mean([r["R_E"] for r in s]):>9.4f}')


if __name__ == '__main__':
    main()
