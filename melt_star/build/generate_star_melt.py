import numpy as np

NSTAR = 20
NARMS = 4
ARM_LEN = 25

BOX = 40.0
BONDLEN = 1.0

atoms = []
bonds = []

atom_id = 1
bond_id = 1
mol_id = 1

for s in range(NSTAR):

    core = np.random.uniform(5, BOX-5, size=3)

    core_id = atom_id

    atoms.append([
        atom_id,
        mol_id,
        1,
        core[0],
        core[1],
        core[2]
    ])

    atom_id += 1

    arm_dirs = np.array([
        [1,0,0],
        [-1,0,0],
        [0,1,0],
        [0,-1,0]
    ])

    for arm in range(NARMS):

        prev = core_id

        for k in range(ARM_LEN):

            pos = core + (k+1)*BONDLEN*arm_dirs[arm]

            pos += np.random.normal(
                scale=0.05,
                size=3
            )

            atoms.append([
                atom_id,
                mol_id,
                1,
                pos[0],
                pos[1],
                pos[2]
            ])

            bonds.append([
                bond_id,
                1,
                prev,
                atom_id
            ])

            prev = atom_id

            atom_id += 1
            bond_id += 1

    mol_id += 1

natoms = len(atoms)
nbonds = len(bonds)

with open("star_melt_init.data","w") as f:

    f.write("Star melt\n\n")

    f.write(f"{natoms} atoms\n")
    f.write(f"{nbonds} bonds\n\n")

    f.write("1 atom types\n")
    f.write("1 bond types\n\n")

    f.write(f"0.0 {BOX} xlo xhi\n")
    f.write(f"0.0 {BOX} ylo yhi\n")
    f.write(f"0.0 {BOX} zlo zhi\n\n")

    f.write("Masses\n\n")
    f.write("1 1.0\n\n")

    f.write("Atoms\n\n")

    for a in atoms:
        f.write(
            f"{a[0]} {a[1]} {a[2]} "
            f"{a[3]:.6f} {a[4]:.6f} {a[5]:.6f}\n"
        )

    f.write("\nBonds\n\n")

    for b in bonds:
        f.write(
            f"{b[0]} {b[1]} {b[2]} {b[3]}\n"
        )

print("Generated star_melt_init.data")
print("Atoms:", natoms)
print("Bonds:", nbonds)
