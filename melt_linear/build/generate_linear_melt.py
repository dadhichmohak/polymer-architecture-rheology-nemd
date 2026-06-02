#!/usr/bin/env python3
"""
generate_linear_melt.py
Generates a Kremer-Grest linear polymer melt initial configuration.

Strategy: Random walk on cubic lattice (spacing = 1.0 sigma)
- Guarantees minimum bead distance = 1.0 sigma (no catastrophic overlaps)
- Large initial box (low density) -- compression handled in LAMMPS equil stage
- Valid LAMMPS data file: units lj, atom_style bond

System: 20 chains x 100 beads = 2000 atoms, 1980 bonds
Output: linear_melt_init.data

Scientific basis:
- Auhl et al. J. Chem. Phys. 119, 12718 (2003)
- Kremer & Grest J. Chem. Phys. 92, 5057 (1990)
"""

import numpy as np
from scipy.spatial import KDTree
import sys
from pathlib import Path

# ============================================================
# Parameters
# ============================================================
N_CHAINS     = 20
N_BEADS      = 100
LATTICE_STEP = 1.0          # sigma — lattice spacing
BOX_L        = 40.0         # sigma — initial box (rho_init = 2000/40^3 = 0.031)
MAX_ATTEMPTS = 500          # attempts to place each bead before backtracking
BACKTRACK    = 20           # beads to backtrack on failure
SEED         = 42
OUTPUT       = "linear_melt_init.data"

rng = np.random.default_rng(SEED)

# 6 cardinal directions on cubic lattice
DIRECTIONS = np.array([
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1]
], dtype=float) * LATTICE_STEP

# ============================================================
# Lattice walk with KDTree overlap check
# ============================================================
def wrap(pos, L):
    """Apply periodic boundary conditions."""
    return pos % L

def build_chain(start, occupied_tree, occupied_list, L,
                max_attempts, backtrack_n, n_beads, rng):
    """
    Grow a single chain bead by bead on the cubic lattice.
    Uses backtracking to escape dead ends.
    Returns array of shape (n_beads, 3) or raises RuntimeError.
    """
    chain = np.zeros((n_beads, 3))
    chain[0] = wrap(start, L)
    placed = 1
    attempts_per_bead = np.zeros(n_beads, dtype=int)

    while placed < n_beads:
        current = chain[placed - 1]
        # Shuffle directions for unbiased walk
        dirs = DIRECTIONS.copy()
        rng.shuffle(dirs)

        placed_this = False
        for d in dirs:
            candidate = wrap(current + d, L)

            # Check against all occupied positions (inter-chain + earlier in this chain)
            all_occupied = occupied_list + chain[:placed].tolist()
            if len(all_occupied) == 0:
                placed_this = True
            else:
                tree = KDTree(all_occupied)
                dist, _ = tree.query(candidate, k=1)
                if dist > LATTICE_STEP * 0.5:   # no site within half a step
                    placed_this = True

            if placed_this:
                chain[placed] = candidate
                placed += 1
                attempts_per_bead[placed - 1] = 0
                break

        if not placed_this:
            attempts_per_bead[placed - 1] += 1
            if attempts_per_bead[placed - 1] > max_attempts:
                # Backtrack
                bt = min(backtrack_n, placed - 1)
                placed -= bt
                attempts_per_bead[placed:placed + bt] = 0
                if placed <= 1:
                    raise RuntimeError(
                        f"Chain growth failed: stuck after placing {placed} beads. "
                        "Try increasing BOX_L or reducing N_CHAINS."
                    )

    return chain

def generate_melt(n_chains, n_beads, L, lattice_step,
                  max_attempts, backtrack_n, rng):
    """Generate all chains."""
    all_chains = []
    occupied = []      # flat list of all placed bead positions

    # Spread starting beads across the box on a coarse grid
    # to avoid all chains starting in the same region
    n_grid = int(np.ceil(n_chains ** (1/3))) + 1
    grid_pts = np.array([
        [i, j, k]
        for i in range(n_grid)
        for j in range(n_grid)
        for k in range(n_grid)
    ], dtype=float)
    grid_pts = (grid_pts / n_grid) * (L * 0.9) + L * 0.05
    rng.shuffle(grid_pts)

    for c in range(n_chains):
        start = grid_pts[c % len(grid_pts)]
        # Snap to lattice
        start = np.round(start / lattice_step) * lattice_step
        start = wrap(start, L)

        print(f"  Growing chain {c+1}/{n_chains} ...", end=" ")
        sys.stdout.flush()

        for attempt in range(5):
            try:
                chain = build_chain(
                    start, occupied, [p for p in occupied],
                    L, max_attempts, backtrack_n, n_beads, rng
                )
                break
            except RuntimeError as e:
                if attempt < 4:
                    # Randomise starting point and retry
                    start = wrap(
                        np.round(rng.uniform(0, L, 3) / lattice_step) * lattice_step,
                        L
                    )
                    print(f"\n    Retrying chain {c+1} with new start ...", end=" ")
                else:
                    raise RuntimeError(
                        f"Chain {c+1} failed after 5 attempts. "
                        "Increase BOX_L."
                    ) from e

        occupied.extend(chain.tolist())
        all_chains.append(chain)
        n_placed = sum(len(ch) for ch in all_chains)
        print(f"OK  ({n_placed}/{n_chains*n_beads} beads total)")

    return np.array(all_chains)   # shape: (n_chains, n_beads, 3)

# ============================================================
# LAMMPS data file writer
# ============================================================
def write_lammps_data(chains, L, filename):
    """Write a valid LAMMPS data file for atom_style bond."""
    n_chains, n_beads, _ = chains.shape
    n_atoms = n_chains * n_beads
    n_bonds = n_chains * (n_beads - 1)

    with open(filename, 'w') as f:
        f.write("LAMMPS data file: Kremer-Grest linear melt\n")
        f.write(f"# {n_chains} chains x {n_beads} beads | "
                f"rho_init = {n_atoms/L**3:.4f} | box = {L}sigma\n\n")

        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_bonds} bonds\n\n")

        f.write("1 atom types\n")
        f.write("1 bond types\n\n")

        f.write(f"0.0 {L:.6f} xlo xhi\n")
        f.write(f"0.0 {L:.6f} ylo yhi\n")
        f.write(f"0.0 {L:.6f} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        f.write("1 1.0\n\n")

        # Atoms: atom_id mol_id atom_type x y z
        f.write("Atoms\n\n")
        atom_id = 1
        for mol_id, chain in enumerate(chains, start=1):
            for bead in chain:
                f.write(f"{atom_id:6d} {mol_id:4d} 1 "
                        f"{bead[0]:12.6f} {bead[1]:12.6f} {bead[2]:12.6f}\n")
                atom_id += 1

        # Bonds: bond_id bond_type atom1 atom2
        f.write("\nBonds\n\n")
        bond_id = 1
        atom_id = 1
        for chain_idx in range(n_chains):
            base = chain_idx * n_beads + 1
            for b in range(n_beads - 1):
                f.write(f"{bond_id:6d} 1 {base+b:6d} {base+b+1:6d}\n")
                bond_id += 1

    print(f"\nData file written: {filename}")
    print(f"  Atoms : {n_atoms}")
    print(f"  Bonds : {n_bonds}")
    print(f"  Box L : {L} sigma")
    print(f"  rho   : {n_atoms/L**3:.5f}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("Kremer-Grest Melt Generator — Lattice Walk Strategy")
    print("=" * 55)
    print(f"  Chains   : {N_CHAINS}")
    print(f"  Beads/ch : {N_BEADS}")
    print(f"  Box L    : {BOX_L} sigma")
    print(f"  rho_init : {N_CHAINS*N_BEADS/BOX_L**3:.4f}")
    print(f"  Lattice  : {LATTICE_STEP} sigma\n")

    chains = generate_melt(
        N_CHAINS, N_BEADS, BOX_L, LATTICE_STEP,
        MAX_ATTEMPTS, BACKTRACK, rng
    )

    write_lammps_data(chains, BOX_L, OUTPUT)
    print("\nDone. Run check_structure.py to validate.")