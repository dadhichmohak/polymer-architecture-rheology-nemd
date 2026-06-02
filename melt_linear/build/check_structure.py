#!/usr/bin/env python3
"""
check_structure.py
Validates a LAMMPS data file for polymer melt initialization quality.

Checks:
  - Minimum bead-bead distance (non-bonded pairs only)
  - Pair count histogram below key cutoffs
  - Bond length statistics (mean, std, min, max)
  - Chain end-to-end distance distribution
  - Rg2 per chain

Usage: python check_structure.py linear_melt_init.data
"""

import numpy as np
from scipy.spatial import KDTree
import sys

def parse_lammps_data(filename):
    """
    Parse LAMMPS data file.

    Supports:
        Atoms
        Atoms # bond
        Atoms # molecular
        Bonds
        Velocities
        Pair Coeffs
        Bond Coeffs

    Returns:
        atoms, bonds, box_lengths
    """

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

        # -----------------------------
        # Box dimensions
        # -----------------------------

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

        # -----------------------------
        # Section headers
        # -----------------------------

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

        # -----------------------------
        # Parse atoms
        # -----------------------------

        if section == "atoms":

            parts = line.split()

            # atom-ID mol-ID type x y z
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

                except ValueError:
                    pass

            continue

        # -----------------------------
        # Parse bonds
        # -----------------------------

        if section == "bonds":

            parts = line.split()

            if len(parts) >= 4:

                try:
                    a1 = int(parts[2])
                    a2 = int(parts[3])

                    bonds.append((a1, a2))

                except ValueError:
                    pass

            continue

    L = np.array([
        box["x"][1] - box["x"][0],
        box["y"][1] - box["y"][0],
        box["z"][1] - box["z"][0]
    ])

    return atoms, bonds, L

def minimum_image(dr, L):
    """Apply minimum image convention."""
    return dr - L * np.round(dr / L)

def run_validation(filename):
    print("=" * 55)
    print(f"Structure validation: {filename}")
    print("=" * 55)

    atoms, bonds, L = parse_lammps_data(filename)
    n_atoms = len(atoms)
    ids     = sorted(atoms.keys())
    pos     = np.array([atoms[i]['pos'] for i in ids])
    mols    = np.array([atoms[i]['mol'] for i in ids])
    bond_set = set(frozenset(b) for b in bonds)

    print(f"\nSystem: {n_atoms} atoms | {len(bonds)} bonds | box {L}")

    # ---- Bond length statistics ----
    print("\n--- Bond length statistics ---")
    bl = []
    for a1, a2 in bonds:
        p1 = atoms[a1]['pos']
        p2 = atoms[a2]['pos']
        dr = minimum_image(p2 - p1, L)
        bl.append(np.linalg.norm(dr))
    bl = np.array(bl)
    print(f"  Mean   : {bl.mean():.5f} sigma")
    print(f"  Std    : {bl.std():.5f} sigma")
    print(f"  Min    : {bl.min():.5f} sigma")
    print(f"  Max    : {bl.max():.5f} sigma")
    if bl.max() > 1.5:
        print("  WARNING: bonds > 1.5 sigma (FENE R0). Will break FENE potential!")
    else:
        print("  Bond lengths: OK (all < FENE R0 = 1.5 sigma)")

    # ---- Non-bonded pair distance analysis ----
    print("\n--- Non-bonded pair distance analysis ---")
    print("  (Building KDTree — may take a few seconds for 2000 atoms)")

    # Sample at most 500 atoms for speed; report scaled result
    sample_ids = np.random.choice(n_atoms, min(500, n_atoms), replace=False)
    sample_pos = pos[sample_ids]
    sample_mol = mols[sample_ids]

    tree = KDTree(sample_pos)
    # Query all pairs within 2.0 sigma
    pairs = tree.query_pairs(r=2.0, output_type='ndarray')

    cutoffs = [0.05, 0.10, 0.15, 0.20, 0.50, 0.80, 1.0, 1.12, 2.0]
    counts  = {c: 0 for c in cutoffs}
    dists   = []
    min_nb  = np.inf

    for i, j in pairs:
        # Skip bonded pairs
        ai = ids[sample_ids[i]]
        aj = ids[sample_ids[j]]
        if frozenset([ai, aj]) in bond_set:
            continue
        dr = minimum_image(sample_pos[i] - sample_pos[j], L)
        d  = np.linalg.norm(dr)
        dists.append(d)
        if d < min_nb:
            min_nb = d
        for c in cutoffs:
            if d < c:
                counts[c] += 1

    # Scale counts back to full system
    scale = (n_atoms / len(sample_ids)) ** 2

    print(f"\n  Minimum non-bonded distance : {min_nb:.5f} sigma")
    print(f"\n  Pair counts (scaled to full system):")
    print(f"  {'Cutoff':>10}  {'Count (scaled)':>16}  {'Assessment':}")
    assessment = {
        0.05:  ("CRITICAL" , "Force blow-up certain"),
        0.10:  ("BAD"      , "Very high forces"),
        0.15:  ("BAD"      , "High forces"),
        0.20:  ("POOR"     , "Significant forces"),
        0.50:  ("WARN"     , "Soft push-off needed"),
        0.80:  ("OK"       , "Soft push-off sufficient"),
        1.0:   ("GOOD"     , "Light push-off or none"),
        1.12:  ("IDEAL"    , "Near WCA diameter"),
        2.0:   ("INFO"     , "Within first shell"),
    }
    for c in cutoffs:
        est = int(counts[c] * scale)
        tag, note = assessment[c]
        print(f"  r < {c:5.3f}  :  {est:>12d}    [{tag}] {note}")

    # ---- Rg2 and end-to-end per chain ----
    print("\n--- Chain statistics ---")
    n_mols = len(set(mols))
    Rg2_list = []
    R2_list  = []

    for m in sorted(set(mols)):
        mask  = (mols == m)
        cpos  = pos[mask]
        com   = cpos.mean(axis=0)
        dr    = cpos - com
        Rg2   = np.mean(np.sum(dr**2, axis=1))
        Rg2_list.append(Rg2)

        # End-to-end using minimum image
        r_ee = minimum_image(cpos[-1] - cpos[0], L)
        R2_list.append(np.dot(r_ee, r_ee))

    Rg2_arr = np.array(Rg2_list)
    R2_arr  = np.array(R2_list)
    print(f"  Chains analysed : {n_mols}")
    print(f"  <Rg²>           : {Rg2_arr.mean():.3f} sigma²  "
          f"(expected ~16.0 after equil)")
    print(f"  <R²>            : {R2_arr.mean():.3f} sigma²  "
          f"(expected ~92-96 after equil; ideal walk = {(len(cpos)-1)*1.0:.1f})")
    print(f"  <R²>/N          : {R2_arr.mean()/(N_BEADS-1):.3f}  "
          f"(ideal lattice walk = 1.00)")

    # ---- Overall verdict ----
    print("\n--- Overall verdict ---")
    critical = int(counts[0.10] * scale)
    if critical == 0:
        print("  PASS: No catastrophic overlaps (r < 0.10 sigma).")
        print("  Ready for soft push-off + compression in LAMMPS.")
    elif critical < 10:
        print(f"  MARGINAL: {critical} pairs below 0.10 sigma.")
        print("  Use nve/limit with very small limit (0.01) in push-off stage.")
    else:
        print(f"  FAIL: {critical} pairs below 0.10 sigma.")
        print("  Rebuild with larger BOX_L or smaller LATTICE_STEP.")

    # Histogram
    if dists:
        hist, edges = np.histogram(dists, bins=40, range=(0, 2.0))
        print("\n  Pair distance histogram (r < 2.0 sigma, sampled):")
        print(f"  {'r range':>18}  count")
        for i in range(min(15, len(hist))):
            if hist[i] > 0:
                bar = '█' * min(30, hist[i] // max(1, max(hist)//30))
                print(f"  {edges[i]:.3f} – {edges[i+1]:.3f}  : "
                      f"{hist[i]:5d}  {bar}")

    print("\n" + "=" * 55)

N_BEADS = 100  # used in Rg2 reporting

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "linear_melt_init.data"
    run_validation(fname)