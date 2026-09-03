"""Emit the manifest of the production set: which runs the paper analyses.

WHY THIS EXISTS. The log archive holds every campaign run for the project,
including exploratory and superseded ones. The paper analyses a subset. Without
a manifest a reader cannot tell which subset, and the run count in the text is
unverifiable. This writes the list.

The production set is the twelve sweeps enumerated in Sec. II.B. Everything
else in the archive is exploratory, superseded, or a tier-3 contact-dump
campaign whose logs are not used for any figure or constraint.

Run it with the archive unpacked alongside, as for the figure scripts:

    tar -xf lammps-logs.tar.xz
    python3 make_manifest.py

Writes derived/manifest.csv and prints the totals that Sec. II.B quotes.
"""
import csv
import glob
import os
import re

# sweep directory -> the description used in the paper's enumeration
PRODUCTION = [
    ('sweep_steady2d',  'friction scan, 2D'),
    ('sweep_steady3d',  'friction scan, 3D'),
    ('sweep_lev_E',     'stiffness lever, 2D'),
    ('sweep_lev_E3d',   'stiffness lever, 3D'),
    ('sweep_lev_P',     'pressure lever'),
    ('sweep_lev_kt',    'tangential stiffness lever'),
    ('sweep_size2d',    'system size lever'),
    ('sweep_restit',    'restitution lever'),
    ('sweep_iscan2',    'shear rates, 2D'),
    ('sweep_iscan3d',   'shear rates, 3D'),
    ('sweep_nothermo2', 'athermal control'),
    ('sweep_velprof',   'velocity profile check'),
]

# log.lammps is LAMMPS' own scratch file, one per working directory, not a run.
SCRATCH = 'log.lammps'

MU = re.compile(r'mu(?P<mu>[0-9.]+)')
TG = re.compile(r'_T(?P<T>[0-9.eE+-]+)')
SD = re.compile(r'_s(?P<s>\d+)$')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows, missing = [], []
    for d, label in PRODUCTION:
        paths = sorted(glob.glob(os.path.join(here, d, 'log.*')))
        paths = [p for p in paths if os.path.basename(p) != SCRATCH]
        if not paths:
            missing.append(d)
            continue
        for p in paths:
            name = os.path.basename(p)
            m, t, s = MU.search(name), TG.search(name), SD.search(name)
            rows.append({
                'sweep': d,
                'role': label,
                'run': name,
                'mu_g': m.group('mu') if m else '',
                'Tgran': t.group('T') if t else '',
                'seed': s.group('s') if s else '',
            })

    if missing:
        print('WARNING: no logs found for ' + ', '.join(missing))
        print('Unpack the log archive in this directory first.')
        if len(missing) == len(PRODUCTION):
            return 1

    out = os.path.join(here, 'derived', 'manifest.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sweep', 'role', 'run',
                                          'mu_g', 'Tgran', 'seed'])
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by[r['sweep']] = by.get(r['sweep'], 0) + 1
    print(f'{"runs":>6}  sweep')
    for d, label in PRODUCTION:
        print(f'{by.get(d, 0):>6}  {d:<18} {label}')
    print(f'{len(rows):>6}  TOTAL, {len(PRODUCTION)} production sweeps')

    every = [p for p in glob.glob(os.path.join(here, 'sweep_*', 'log.*'))
             if os.path.basename(p) != SCRATCH]
    print(f'\n{len(every)} run logs in the archive overall, across '
          f'{len(glob.glob(os.path.join(here, "sweep_*")))} sweep directories.')
    print(f'The remaining {len(every) - len(rows)} are exploratory, superseded,')
    print('or tier-3 contact-dump campaigns not used for any figure.')
    print(f'\nwritten -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
