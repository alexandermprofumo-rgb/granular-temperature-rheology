"""
Generate a polydisperse 3D sphere packing at target volume fraction phi and
write it as a LAMMPS data file (atom_style sphere) for in.granular_3d.

Usage:
    python3 gen_data_3d.py --N 4000 --phi 0.60 --seed 1 --out data.granular3d
"""
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=4000)
ap.add_argument('--phi', type=float, default=0.60)
ap.add_argument('--rlo', type=float, default=0.35)
ap.add_argument('--rhi', type=float, default=0.65)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='data.granular3d')
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
N = args.N
r = rng.uniform(args.rlo, args.rhi, size=N)
vol = np.sum((4.0/3.0) * np.pi * r**3)
L = (vol / args.phi) ** (1.0/3.0)

n_side = int(np.ceil(N ** (1.0/3.0)))
xs, ys, zs = np.meshgrid(np.linspace(0, L, n_side, endpoint=False),
                          np.linspace(0, L, n_side, endpoint=False),
                          np.linspace(0, L, n_side, endpoint=False))
pos = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)[:N]
pos += rng.uniform(-0.05, 0.05, size=pos.shape) * (L / n_side)
pos %= L

density = 1.0

with open(args.out, 'w') as f:
    f.write("LAMMPS data file: 3D polydisperse granular packing\n\n")
    f.write(f"{N} atoms\n")
    f.write("1 atom types\n\n")
    f.write(f"0.0 {L} xlo xhi\n")
    f.write(f"0.0 {L} ylo yhi\n")
    f.write(f"0.0 {L} zlo zhi\n")
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms # sphere\n\n")
    for i in range(N):
        f.write(f"{i+1} 1 {2*r[i]:.6f} {density:.4f} "
                f"{pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f}\n")
    f.write("\nVelocities\n\n")
    for i in range(N):
        f.write(f"{i+1} 0.0 0.0 0.0 0.0 0.0 0.0\n")

print(f"Wrote {args.out}: N={N}, L={L:.4f}, phi={args.phi}, "
      f"mean diameter={2*np.mean(r):.4f}")
