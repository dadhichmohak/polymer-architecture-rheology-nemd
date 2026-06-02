#!/usr/bin/env python3

import numpy as np
from scipy.spatial import KDTree
import sys

def parse_lammps_data(filename):

    atoms = {}
    bonds = []
    box = {}

    with open(filename) as f:
        lines = f.readlines()

    section = None

    for raw in lines:

        line = raw.strip()

        if not line:
            continue

        if "xlo xhi" in line:
            lo, hi = map(float, line.split()[:2])
            box["x"] = (lo, hi)
            continue

        if "ylo yhi" in line:
            lo, hi = map(float, line.split()[:2])
            box["y"] = (lo, hi)
            continue

        if "zlo zhi" in line:
            lo, hi = map(float, line.split()[:2])
            box["z"] = (lo, hi)
            continue

        if line.startswith("Atoms"):
            section = "atoms"
            continue

        if line.startswith("Bonds"):
            section = "bonds"
            continue

        if (
            line.startswith("Masses")
            or line.startswith("Velocities")
            or line.startswith("Pair Coeffs")
            or line.startswith("Bond Coeffs")
            or line.startswith("Angle Coeffs")
        ):
            section = None
            continue

        if section == "atoms":

            parts = line.split()

            if len(parts) >= 6:

                try:
                    aid = int(parts[0])
                    mol = int(parts[1])

                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])

                    atoms[aid] = {
                        "mol": mol,
                        "pos": np.array([x, y, z])
                    }

                except:
                    pass

            continue

        if section == "bonds":

            parts = line.split()

            if len(parts) >= 4:

                try:
                    a1 = int(parts[2])
                    a2 = int(parts[3])

                    bonds.append((a1, a2))

                except:
                    pass

            continue

    L = np.array([
        box["x"][1] - box["x"][0],
        box["y"][1] - box["y"][0],
        box["z"][1] - box["z"][0]
    ])

    return atoms, bonds, L


def minimum_image(dr, L):
    return dr - L * np.round(dr / L)


def run_validation(filename):

    print("=" * 60)
    print("STAR MELT STRUCTURE VALIDATION")
    print("=" * 60)

    atoms, bonds, L = parse_lammps_data(filename)

    ids = sorted(atoms.keys())

    pos = np.array([atoms[i]["pos"] for i in ids])
    mols = np.array([atoms[i]["mol"] for i in ids])

    n_atoms = len(atoms)
    n_bonds = len(bonds)
    n_mols = len(set(mols))

    beads_per_polymer = n_atoms // n_mols

    print()
    print(f"Atoms             : {n_atoms}")
    print(f"Bonds             : {n_bonds}")
    print(f"Polymers          : {n_mols}")
    print(f"Beads/polymer     : {beads_per_polymer}")
    print(f"Box               : {L}")

    density = n_atoms / np.prod(L)

    print(f"Number density    : {density:.5f}")

    bond_set = set(frozenset(b) for b in bonds)

    print("\n--- Bond Statistics ---")

    bl = []

    for a1, a2 in bonds:

        p1 = atoms[a1]["pos"]
        p2 = atoms[a2]["pos"]

        dr = minimum_image(p2 - p1, L)

        bl.append(np.linalg.norm(dr))

    bl = np.array(bl)

    print(f"Mean bond length  : {bl.mean():.5f}")
    print(f"Std bond length   : {bl.std():.5f}")
    print(f"Min bond length   : {bl.min():.5f}")
    print(f"Max bond length   : {bl.max():.5f}")

    if bl.max() > 1.5:
        print("WARNING: Bond exceeds FENE R0 = 1.5")
    else:
        print("Bond lengths OK")

    print("\n--- Overlap Analysis ---")

    sample_size = min(500, n_atoms)

    sample_ids = np.random.choice(
        n_atoms,
        sample_size,
        replace=False
    )

    sample_pos = pos[sample_ids]

    tree = KDTree(sample_pos)

    pairs = tree.query_pairs(
        r=2.0,
        output_type="ndarray"
    )

    min_nb = 1e9

    counts = {
        0.05: 0,
        0.10: 0,
        0.15: 0,
        0.20: 0,
        0.50: 0
    }

    for i, j in pairs:

        ai = ids[sample_ids[i]]
        aj = ids[sample_ids[j]]

        if frozenset([ai, aj]) in bond_set:
            continue

        dr = minimum_image(
            sample_pos[i] - sample_pos[j],
            L
        )

        d = np.linalg.norm(dr)

        min_nb = min(min_nb, d)

        for c in counts:
            if d < c:
                counts[c] += 1

    scale = (n_atoms / sample_size) ** 2

    print(f"Minimum nonbonded distance : {min_nb:.5f}")

    for c in counts:

        est = int(counts[c] * scale)

        print(
            f"Pairs r<{c:4.2f} : {est}"
        )

    print("\n--- Polymer Size Statistics ---")

    rg2_list = []

    for mol in sorted(set(mols)):

        mask = (mols == mol)

        cpos = pos[mask]

        com = cpos.mean(axis=0)

        dr = cpos - com

        rg2 = np.mean(
            np.sum(dr * dr, axis=1)
        )

        rg2_list.append(rg2)

    rg2_list = np.array(rg2_list)

    print(f"<Rg²>             : {rg2_list.mean():.3f}")
    print(f"Std(Rg²)          : {rg2_list.std():.3f}")

    print("\n--- Verdict ---")

    critical = int(counts[0.10] * scale)

    if critical == 0:
        print("PASS")
        print("Ready for Stage A push-off.")
    elif critical < 10:
        print("MARGINAL")
        print("Use conservative nve/limit.")
    else:
        print("FAIL")
        print("Rebuild initial melt.")

    print("\n" + "=" * 60)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "star_melt_init.data"

    run_validation(fname)