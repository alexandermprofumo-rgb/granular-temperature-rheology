"""Emit the manifest of the production set: which runs the paper analyses.

WHY THIS EXISTS. The log archive holds every campaign run for the project,
including exploratory and superseded ones. The paper analyses a subset. Without
a manifest a reader cannot tell which subset, and the run count in the text is
unverifiable. This writes the list.

The production set is the sweeps enumerated in Sec. II.B. Everything else in
the archive is exploratory, superseded, or a tier-3 contact-dump campaign whose
logs are not used for any figure or constraint.

A RUN COUNTS ONLY IF IT FINISHED. An earlier version counted every log file in
a production directory, which put two runs into the published total that never
produced data: the frictionless athermal cells at restitution 0.9 stop at step
zero with "Too many neighbor bins", because without a bath the packing does not
stay dense and the barostat inflates the box. Logs without the DONE marker are
now listed separately rather than counted.

THREE COUNTS DESCRIBE THE SAME ARCHIVE, and every one of them has been quoted
somewhere. They are printed together here with their definitions so that the
paper, the README and the data deposit can quote one of them deliberately:
files in the tarball, run logs across all directories, and run logs inside
sweep_* directories.

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
    ('sweep_iscan_stiff',      'shear rates at ten times the modulus, 2D'),
    ('sweep_iscan_stiff_E1e4', 'shear rates at one tenth the modulus, 2D'),
]

# log.lammps is LAMMPS' own scratch file, one per working directory, not a run.
SCRATCH = 'log.lammps'

MU = re.compile(r'mu(?P<mu>[0-9.]+)')
TG = re.compile(r'_T(?P<T>[0-9.eE+-]+)')
SD = re.compile(r'_s(?P<s>\d+)$')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows, missing, unfinished = [], [], []
    for d, label in PRODUCTION:
        paths = sorted(glob.glob(os.path.join(here, d, 'log.*')))
        paths = [p for p in paths if os.path.basename(p) != SCRATCH]
        if not paths:
            missing.append(d)
            continue
        for p in paths:
            with open(p, errors='ignore') as fh:
                if not any(l.startswith('DONE') for l in fh):
                    unfinished.append((d, os.path.basename(p)))
                    continue
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
    live = sum(1 for d, _ in PRODUCTION if by.get(d))
    print(f'{len(rows):>6}  TOTAL, finished runs in {live} production sweeps')
    if unfinished:
        print(f'\n{len(unfinished)} production logs never finished and are NOT counted:')
        for d, name in unfinished:
            print(f'        {d}/{name}')

    dirs = [x for x in glob.glob(os.path.join(here, '*')) if os.path.isdir(x)
            and glob.glob(os.path.join(x, 'log.*'))]
    files = [p for x in dirs for p in glob.glob(os.path.join(x, 'log.*'))]
    runs_all = [p for p in files if os.path.basename(p) != SCRATCH]
    sweeps = [x for x in dirs if os.path.basename(x).startswith('sweep_')]
    runs_sw = [p for p in runs_all
               if os.path.basename(os.path.dirname(p)).startswith('sweep_')]
    print('\nTHE ARCHIVE, THREE WAYS')
    print(f'  {len(files):>6}  log files of any kind, in {len(dirs)} directories'
          '   (what tar lists)')
    print(f'  {len(runs_all):>6}  run logs, excluding LAMMPS scratch log.lammps')
    print(f'  {len(runs_sw):>6}  run logs inside the {len(sweeps)} sweep_* directories')
    print(f'The {len(runs_all) - len(rows)} run logs outside the production set are')
    print('exploratory, superseded, unfinished, or tier-3 contact-dump campaigns.')
    print(f'\nwritten -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
