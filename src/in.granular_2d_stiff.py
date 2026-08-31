"""
Helper for run_stiffness.sh: timestep and step counts for a given modulus.

The Hertzian collision time scales as t_c ~ E^(-2/5), so holding the same
PHYSICAL duration (and hence the same accumulated strain, since gdot is fixed)
while changing E requires both the timestep and the step count to move:

    dt(E)     = dt0 * (E0/E)^(2/5)
    nsteps(E) = nsteps0 * dt0/dt(E)

Without this the stiff runs would be integrated too coarsely to be stable and
the soft runs would cover a different strain interval than the reference, so
any change in n could not be attributed to the contact timescale.

Usage:  python3 in.granular_2d_stiff.py <E>   ->  "dt nrelax nequil nmeas"
"""
import sys

E0, DT0 = 1.0e5, 0.001
NRELAX0, NEQUIL0, NMEAS0 = 3000, 40000, 80000

E = float(sys.argv[1])
dt = DT0 * (E0 / E) ** 0.4
scale = DT0 / dt
print(f"{dt:.8g} {round(NRELAX0*scale)} {round(NEQUIL0*scale)} {round(NMEAS0*scale)}")
