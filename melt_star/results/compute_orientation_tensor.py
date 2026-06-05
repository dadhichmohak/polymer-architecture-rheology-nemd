import numpy as np

# ============================================================
# User inputs
# ============================================================

DATAFILE = "../equil/output/star_melt_extended.data"
TRAJFILE = "../nemd/gdot_0.1/output/traj_nemd.lammpstrj"

# ============================================================
# Read bonds from LAMMPS data file
# ============================================================

bonds = []

with open(DATAFILE) as f:

    lines = f.readlines()

start = None

for i, line in enumerate(lines):

    if line.strip() == "Bonds":
        start = i + 2
        break

if start is None:
    raise RuntimeError("Could not find Bonds section")

for line in lines[start:]:

    s = line.strip()

    if not s:
        continue

    parts = s.split()

    if len(parts) != 4:
        break

    _, _, a1, a2 = map(int, parts)

    bonds.append((a1 - 1, a2 - 1))

print(f"Read {len(bonds)} bonds")

# ============================================================
# Read trajectory
# ============================================================

frames = []

with open(TRAJFILE) as f:

    while True:

        line = f.readline()

        if not line:
            break

        if "ITEM: TIMESTEP" not in line:
            continue

        f.readline()  # timestep

        f.readline()  # NUMBER OF ATOMS
        natoms = int(f.readline())

        f.readline()  # BOX BOUNDS

        xline = f.readline().split()
        yline = f.readline().split()
        zline = f.readline().split()

        Lx = float(xline[1]) - float(xline[0])
        Ly = float(yline[1]) - float(yline[0])
        Lz = float(zline[1]) - float(zline[0])

        f.readline()  # ITEM: ATOMS

        atoms = np.zeros((natoms, 3))

        for _ in range(natoms):

            vals = f.readline().split()

            aid = int(vals[0])

            x = float(vals[3])
            y = float(vals[4])
            z = float(vals[5])

            atoms[aid - 1] = [x, y, z]

        frames.append(atoms)

print(f"Frames = {len(frames)}")

# ============================================================
# Orientation tensor
# ============================================================

S = np.zeros((3, 3))
nbond_vectors = 0

for atoms in frames:

    for i, j in bonds:

        r = atoms[j] - atoms[i]

        norm = np.linalg.norm(r)

        if norm < 1e-12:
            continue

        u = r / norm

        S += 0.5 * (3.0 * np.outer(u, u) - np.eye(3))

        nbond_vectors += 1

S /= nbond_vectors

eigvals, eigvecs = np.linalg.eigh(S)

imax = np.argmax(eigvals)

# ============================================================
# Output
# ============================================================

with open("orientation_star.txt", "w") as f:

    f.write(f"Sxx = {S[0,0]:.6f}\n")
    f.write(f"Syy = {S[1,1]:.6f}\n")
    f.write(f"Szz = {S[2,2]:.6f}\n\n")

    f.write(f"Sxy = {S[0,1]:.6f}\n")
    f.write(f"Sxz = {S[0,2]:.6f}\n")
    f.write(f"Syz = {S[1,2]:.6f}\n\n")

    f.write("Eigenvalues:\n")

    for val in eigvals:
        f.write(f"{val:.6f}\n")

    f.write("\nPrincipal direction:\n")

    vec = eigvecs[:, imax]

    for x in vec:
        f.write(f"{x:.6f}\n")

print("Orientation tensor written to orientation_star.txt")