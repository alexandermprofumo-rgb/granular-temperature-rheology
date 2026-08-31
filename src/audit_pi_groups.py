"""
Dimensionless-parameter audit of every sweep in the project.

WHY THIS EXISTS. Four conclusions in this campaign were overturned by a
variable nobody was tracking: pressure (produced the spurious "framework
swap", inflated the force-weighted fabric result, and reversed the k_t sign
structure), accumulated strain, and the inertial number. Each was in principle
predictable from a parameter list that was never written down. This writes it
down and checks every sweep against it, so confounds are found BEFORE a
conclusion is drawn rather than after.

THE COMPLETE GROUP LIST. With grains of mean diameter d and density rho under
confining pressure P, sheared at gdot, with Hertzian modulus E, tangential
stiffness k_t, restitution e, friction mu_g, and a Langevin bath of damping
time t_damp, the independent dimensionless groups are:

    I        = gdot * d * sqrt(rho/P)      inertial number
    Theta    = rho <dv^2> / P              the scan variable (dimensionless already)
    kappa    = E / P                       grain stiffness / softness
    k_t/k_n  tangential stiffness ratio
    mu_g     grain friction
    e        restitution
    Pi_damp  = t_damp * sqrt(P/rho) / d    THERMOSTAT COUPLING
    D        dimension
    N        particle count
    nu       Poisson ratio        (0.3 in every run)
    poly     size polydispersity  (r in [0.35,0.65] in every run)

Pi_damp is the one that was never considered. t_damp = 0.5 is hardcoded in
ABSOLUTE time units in every input file, but the microscopic inertial time is
t_p = d*sqrt(rho/P), which depends on pressure. So changing Pconf silently
changes the thermostat coupling -- and Tier 0-A already established that n
depends on it. Any comparison across pressure therefore moves at least three
groups at once (I, kappa, Pi_damp), never one.

A comparison is only interpretable if the two states differ in exactly ONE
group. This script reports, for every pair of sweeps, how many groups differ.

Usage:  python3 audit_pi_groups.py
"""
import numpy as np

D_MEAN, RHO, TDAMP, NU = 1.0, 1.0, 0.5, 0.3
KN_PREFAC, KT_DEFAULT = 7.326e4, 9.05e4
KTKN_DEFAULT = KT_DEFAULT / KN_PREFAC


def groups(P, gdot, E, ktkn, D, N, e=0.5, thermostat=True):
    return dict(
        I=gdot * D_MEAN * np.sqrt(RHO / P),
        kappa=E / P,
        kt_kn=ktkn,
        e=e,
        Pi_damp=(TDAMP * np.sqrt(P / RHO) / D_MEAN) if thermostat else np.inf,
        D=float(D),
        N=float(N),
    )


# (name, fixed groups, which groups the sweep deliberately SCANS)
SWEEPS = [
    ('production2D_P2', groups(2.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g']),
    ('matched2D_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g']),
    ('3D_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 3, 4000), ['mu_g']),
    ('stiffness_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g', 'kappa']),
    ('kt_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g', 'kt_kn']),
    ('ktgrid_P2', groups(2.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g', 'kt_kn']),
    ('nofI_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g', 'I']),
    ('nothermo_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000,
                            thermostat=False), ['mu_g', 'e']),
    ('steadystate_P10', groups(10.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 4000), ['mu_g']),
    ('size_P2', groups(2.0, 1.0e-3, 1.0e5, KTKN_DEFAULT, 2, 2000), ['mu_g', 'N']),
]

KEYS = ['I', 'kappa', 'kt_kn', 'e', 'Pi_damp', 'D', 'N']


def main():
    print('BASELINE STATE OF EVERY SWEEP (values at the centre of each scan)\n')
    hdr = f'{"sweep":<18}' + ''.join(f'{k:>11}' for k in KEYS) + '   scans'
    print(hdr)
    print('-' * len(hdr))
    for name, g, scan in SWEEPS:
        row = f'{name:<18}'
        for k in KEYS:
            v = g[k]
            row += f'{v:>11.4g}' if np.isfinite(v) else f'{"none":>11}'
        print(row + '   ' + ','.join(scan))

    print('\n\nPAIRWISE COMPARABILITY  (groups differing, excluding each sweep\'s own scan)')
    print('a comparison is clean only if EXACTLY ONE group differs\n')
    n = len(SWEEPS)
    clean, confounded = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ni, gi, si = SWEEPS[i]
            nj, gj, sj = SWEEPS[j]
            diff = []
            for k in KEYS:
                a, b = gi[k], gj[k]
                if not (np.isfinite(a) and np.isfinite(b)):
                    if a != b:
                        diff.append(k)
                    continue
                if abs(a - b) > 1e-9 * max(abs(a), abs(b), 1.0):
                    diff.append(k)
            (clean if len(diff) == 1 else confounded).append((ni, nj, diff))

    print(f'CLEAN ({len(clean)} pairs) -- one group apart, directly interpretable:')
    for a, b, d in clean:
        print(f'   {a:<18} vs {b:<18} differ in: {d[0]}')

    print(f'\nCONFOUNDED ({len(confounded)} pairs) -- showing those actually used '
          'for a conclusion:')
    flagged = {('production2D_P2', 'matched2D_P10'),
               ('ktgrid_P2', 'kt_P10'),
               ('production2D_P2', '3D_P10'),
               ('ktgrid_P2', 'stiffness_P10')}
    for a, b, d in confounded:
        if (a, b) in flagged or (b, a) in flagged:
            print(f'   {a:<18} vs {b:<18} differ in {len(d)}: {", ".join(d)}')

    print("""
WHAT THIS VOIDS
---------------
Every Pconf=2 vs Pconf=10 comparison moves THREE groups at once: I (2.24x),
kappa (5x) and Pi_damp (2.24x). So these claims are NOT established:

  * "kappa = E/P is the wrong variable, pressure enters separately"
    (the 4-7 sigma extrapolation failure). The P=2 rows also differ in I and
    in thermostat coupling, and Tier 0-A showed n depends on the latter. The
    failure is real but its CAUSE is unidentified.

  * the size of the P=2 vs P=10 differences in the k_t sign structure, the
    force-weighted fabric advantage, and the framework swap. That these were
    pressure-SENSITIVE stands; which group did it does not.

WHAT SURVIVES
-------------
Everything measured WITHIN a single sweep at fixed Pconf=10 -- the friction
curve, the dimension ratio A_3/A_2, both stiffness derivatives, the two-lever
collapse refutation, the steady-state and I-independence checks. Those never
cross a pressure boundary.

THE FIX
-------
Scale t_damp with pressure as t_damp = Pi_damp * d * sqrt(rho/P) to hold the
thermostat coupling fixed, and scale gdot as sqrt(P) to hold I fixed. Then a
pressure scan moves kappa alone. Note the ORIGINAL pressure lever failed
because gdot ~ sqrt(P) drove shear heating past the thermostat -- holding
Pi_damp fixed rather than t_damp is exactly what was missing there, since a
stiffer bath at higher P is what keeps control.
""")


if __name__ == '__main__':
    main()
