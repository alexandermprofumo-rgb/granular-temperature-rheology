"""What each set of simulations varies, and over what range.

WHY THIS EXISTS.  Sec. II names the sets of simulations and Table I names the
control parameters, but the ranges were quoted from memory in the text.  This
reads them off the run filenames instead, so the table in Sec. II.C can be
regenerated and checked rather than retyped.

Every run is named by the parameters that distinguish it, for example
log.E1.0e4_mu0.1_T0.0005_s1, so the distinct values of each key give the range
actually run.  T is the target temperature and s the packing, which every set
varies; they are reported once rather than per row.

Usage:  python3 scan_summary.py       (run from the directory holding the sweeps)
"""
import glob
import os
import re

KEY = {'mu': 'grain friction $\\mug$', 'E': 'modulus $E$',
       'P': 'pressure $P$', 'k': 'tangential stiffness $k_t$',
       'N': 'grain count $N$', 'e': 'restitution $e$',
       'g': 'shear rate $\\dot\\gamma$', 'td': 'bath damping time',
       'T': 'target temperature', 's': 'packing'}

SETS = [
    ('sweep_steady2d',  'friction scan, 2D'),
    ('sweep_steady3d',  'friction scan, 3D'),
    ('sweep_lev_E',     'stiffness scan, 2D'),
    ('sweep_lev_E3d',   'stiffness scan, 3D'),
    ('sweep_lev_P',     'pressure scan'),
    ('sweep_lev_kt',    'tangential stiffness scan'),
    ('sweep_size2d',    'system size scan'),
    ('sweep_restit',    'restitution scan'),
    ('sweep_lev_td',    'bath damping scan'),
    ('sweep_iscan2',    'shear rates, 2D'),
    ('sweep_iscan3d',   'shear rates, 3D'),
    ('sweep_iscan_stiff',      'shear rates at ten times the modulus, 2D'),
    ('sweep_iscan_stiff_E1e4', 'shear rates at one tenth the modulus, 2D'),
    ('sweep_iscan_stiff_E1e7', 'shear rates at a hundred times the modulus, 2D'),
    ('sweep_nothermo2', 'athermal control'),
    ('sweep_velprof',   'velocity-profile check'),
]

FIELD = re.compile(r'^(td|mu|nt|[EPkNegTs])(-?[0-9][0-9.eE+-]*)$')


def values(d):
    out = {}
    for p in glob.glob(os.path.join(d, 'log.*')):
        b = os.path.basename(p)
        if b == 'log.lammps':
            continue
        for f in b[len('log.'):].split('_'):
            m = FIELD.match(f)
            if not m:
                continue
            try:
                out.setdefault(m.group(1), set()).add(float(m.group(2)))
            except ValueError:
                pass
    return out


def span(vs):
    v = sorted(vs)
    if len(v) == 1:
        return f'{v[0]:g}'
    return f'{v[0]:g} to {v[-1]:g} ({len(v)} values)'


def main():
    print(f'{"set of simulations":42}{"runs":>6}  varies')
    total = 0
    for d, label in SETS:
        if not os.path.isdir(d):
            print(f'{label:42}{"--":>6}  directory not found')
            continue
        n = len([p for p in glob.glob(os.path.join(d, 'log.*'))
                 if os.path.basename(p) != 'log.lammps'])
        total += n
        vs = values(d)
        parts = [f'{KEY.get(k, k)}: {span(v)}'
                 for k, v in vs.items() if k not in ('T', 's') and len(v) > 1]
        fixed = [f'{KEY.get(k, k)} = {span(v)}'
                 for k, v in vs.items() if k not in ('T', 's') and len(v) == 1]
        temps = span(vs.get('T', {0}))
        seeds = len(vs.get('s', {0}))
        print(f'{label:42}{n:>6}  ' + ('; '.join(parts) if parts else 'nothing'))
        print(f'{"":48}  fixed: ' + ('; '.join(fixed) if fixed else 'none'))
        print(f'{"":48}  targets: {temps}, packings: {seeds}')
    print(f'{"":42}{total:>6}  log files in {len(SETS)} sets')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
