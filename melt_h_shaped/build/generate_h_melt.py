import numpy as np

NPOLY = 20

BACKBONE = 51
ARM_LEN = 25

BOX = 80.0
BONDLEN = 1.0

atoms = []
bonds = []

atom_id = 1
bond_id = 1
mol_id = 1

for p in range(NPOLY):

    center = np.random.uniform(8, BOX-8, size=3)

    backbone_ids = []

    # --------------------------------------------------
    # Backbone
    # --------------------------------------------------

    start = center - np.array([25.0,0.0,0.0])

    prev = None

    for k in range(BACKBONE):

        pos = start + np.array([k*BONDLEN,0,0])

        pos += np.random.normal(scale=0.05,size=3)

        atoms.append([
            atom_id,
            mol_id,
            1,
            pos[0],
            pos[1],
            pos[2]
        ])

        backbone_ids.append(atom_id)

        if prev is not None:

            bonds.append([
                bond_id,
                1,
                prev,
                atom_id
            ])

            bond_id += 1

        prev = atom_id
        atom_id += 1

    # --------------------------------------------------
    # Side arms
    # --------------------------------------------------

    middle = backbone_ids[BACKBONE//2]

    middle_pos = np.array(
        atoms[middle-1][3:6]
    )

    # Upper arm (+y)

    prev = middle

    for k in range(ARM_LEN):

        pos = middle_pos + np.array(
            [0,(k+1)*BONDLEN,0]
        )

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

    # Lower arm (-y)

    prev = middle

    for k in range(ARM_LEN):

        pos = middle_pos + np.array(
            [0,-(k+1)*BONDLEN,0]
        )

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

with open("h_melt_init.data","w") as f:

    f.write("H-shaped melt\n\n")

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

print("H-shaped melt generated")
print(f"Atoms : {natoms}")
print(f"Bonds : {nbonds}")
