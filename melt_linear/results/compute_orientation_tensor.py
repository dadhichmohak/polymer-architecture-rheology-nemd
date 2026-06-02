import numpy as np

traj = "../nemd/gdot_0.1/output/traj_nemd.lammpstrj"

NCHAIN = 20
NBEADS = 100

frames = []

with open(traj) as f:
    while True:
        line = f.readline()
        if not line:
            break

        if "ITEM: TIMESTEP" not in line:
            continue

        timestep = int(f.readline())

        f.readline()          # NUMBER OF ATOMS
        natoms = int(f.readline())

        f.readline()          # BOX BOUNDS
        xline = f.readline().split()
        yline = f.readline().split()
        zline = f.readline().split()

        Lx = float(xline[1]) - float(xline[0])
        Ly = float(yline[1]) - float(yline[0])
        Lz = float(zline[1]) - float(zline[0])

        f.readline()          # ITEM: ATOMS

        atoms = np.zeros((natoms,3))

        for _ in range(natoms):
            vals = f.readline().split()
            aid = int(vals[0])
            x,y,z = map(float,vals[3:6])
            atoms[aid-1] = [x,y,z]

        frames.append(atoms)

print("Frames =",len(frames))

S = np.zeros((3,3))
nbonds = 0

for atoms in frames:

    for c in range(NCHAIN):

        start = c*NBEADS

        for i in range(NBEADS-1):

            r1 = atoms[start+i]
            r2 = atoms[start+i+1]

            u = r2-r1

            norm = np.linalg.norm(u)

            if norm < 1e-12:
                continue

            u /= norm

            S += 0.5*(3*np.outer(u,u)-np.eye(3))

            nbonds += 1

S /= nbonds

print("\nOrientation tensor Sij\n")
print(S)

eigvals,eigvecs=np.linalg.eigh(S)

print("\nEigenvalues:")
print(eigvals)

print("\nPrincipal orientation:")
print(eigvecs[:,np.argmax(eigvals)])
