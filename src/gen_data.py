"""
Generate a polydisperse 2D disk packing at target area fraction phi and
write it as a LAMMPS data file (atom_style sphere) for in.granular_2d.

Usage:
    python3 gen_data.py --N 4000 --phi 0.80 --seed 1 --out data.granular
"""
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=4000)
ap.add_argument('--phi', type=float, default=0.80)
ap.add_argument('--rlo', type=float, default=0.35)
ap.add_argument('--rhi', type=float, default=0.65)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='data.granular')
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
N = args.N
r = rng.uniform(args.rlo, args.rhi, size=N)
area = np.sum(np.pi * r**2)
L = np.sqrt(area / args.phi)

# place on a jittered grid (avoids large initial overlaps; the LAMMPS run
# does a brief soft-push relaxation afterward anyway)
n_side = int(np.ceil(np.sqrt(N)))
xs, ys = np.meshgrid(np.linspace(0, L, n_side, endpoint=False),
                      np.linspace(0, L, n_side, endpoint=False))
pos = np.stack([xs.ravel(), ys.ravel()], axis=1)[:N]
pos += rng.uniform(-0.05, 0.05, size=pos.shape) * (L / n_side)
pos[:, 0] %= L
pos[:, 1] %= L

density = 1.0
mass = density * np.pi * r**2

with open(args.out, 'w') as f:
    f.write("LAMMPS data file: 2D polydisperse granular packing\n\n")
    f.write(f"{N} atoms\n")
    f.write("1 atom types\n\n")
    f.write(f"0.0 {L} xlo xhi\n")
    f.write(f"0.0 {L} ylo yhi\n")
    f.write("-0.5 0.5 zlo zhi\n")
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # sphere\n\n")
    for i in range(N):
        # id type diameter density x y z
        f.write(f"{i+1} 1 {2*r[i]:.6f} {density:.4f} "
                f"{pos[i,0]:.6f} {pos[i,1]:.6f} 0.0\n")
    f.write("\nVelocities\n\n")
    for i in range(N):
        f.write(f"{i+1} 0.0 0.0 0.0 0.0 0.0 0.0\n")

print(f"Wrote {args.out}: N={N}, L={L:.4f}, phi={args.phi}, "
      f"mean diameter={2*np.mean(r):.4f}")
