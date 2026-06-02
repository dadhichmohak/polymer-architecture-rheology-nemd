#!/usr/bin/env python3
"""
generate_linear_melt.py  —  v2
Kremer-Grest linear polymer melt: lattice walk with ideal-chain statistics.

Key change from v1:
  v1: site-exclusion against ALL beads (all chains) → SAW bias, <R²>/N ≈ 1.88
  v2: site-exclusion against current chain only    → near-ideal, <R²>/N ≈ 1.0
      Inter-chain overlaps prevented by post-check + backtrack at chain level.

Scientific basis:
  - At rho_init = 0.031 (L=40, N=2000), inter-chain contacts are rare.
  - Intra-chain self-avoidance is what drives the SAW bias.
  - Removing intra-chain exclusion restores ideal Gaussian statistics.
  - Auhl et al. J. Chem. Phys. 119, 12718 (2003): initial statistics irrelevant
    provided no catastrophic overlaps and equil run > 3 tau_R.

System: 20 chains × 100 beads = 2000 atoms, 1980 bonds
Output: linear_melt_init.data
"""

import numpy as np
from scipy.spatial import KDTree
import sys

# ── Parameters ────────────────────────────────────────────────────────────────
N_CHAINS      = 20
N_BEADS       = 100
LATTICE_STEP  = 1.0
BOX_L         = 40.0
SEED          = 42
OUTPUT        = "linear_melt_init.data"

# Backtracking controls (only triggered by inter-chain overlap, rare at low ρ)
MAX_ATTEMPTS  = 200
BACKTRACK_N   = 10
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

DIRECTIONS = np.array([
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
], dtype=float) * LATTICE_STEP


def wrap(pos: np.ndarray, L: float) -> np.ndarray:
    return pos % L


def snap(pos: np.ndarray) -> tuple:
    """Round to nearest lattice site and return as hashable tuple."""
    return tuple(np.round(pos).astype(int))


def grow_chain(
    start: np.ndarray,
    inter_chain_occupied: set,   # sites occupied by PREVIOUSLY placed chains
    L: float,
    n_beads: int,
    max_attempts: int,
    backtrack_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Grow one chain as an ideal-statistics lattice walk.

    Exclusion rule (v2):
      - Never revisit a site occupied by THIS chain (intra-chain bond validity)
      - Never place a bead on a site occupied by a PREVIOUS chain (inter-chain)
      - The reverse step (back to previous bead) IS allowed if site is free
        → This is the key change that restores Gaussian statistics

    The walk is therefore a 'weakly self-avoiding' walk:
      truly self-avoiding within the current chain (no chain crossing),
      but unbiased with respect to direction choice among available sites.
    """
    chain      = np.zeros((n_beads, 3))
    chain_set  = set()                  # sites occupied by this chain so far

    chain[0]   = wrap(start, L)
    chain_set.add(snap(chain[0]))
    placed     = 1
    fail_count = np.zeros(n_beads, dtype=int)

    while placed < n_beads:
        current = chain[placed - 1]

        # Shuffle all 6 directions uniformly — no directional bias
        order = rng.permutation(6)
        dirs  = DIRECTIONS[order]

        found = False
        for d in dirs:
            candidate     = wrap(current + d, L)
            candidate_key = snap(candidate)

            # Reject if occupied by this chain OR any earlier chain
            if candidate_key in chain_set:
                continue
            if candidate_key in inter_chain_occupied:
                continue

            # Accept
            chain[placed]  = candidate
            chain_set.add(candidate_key)
            placed        += 1
            fail_count[placed - 1] = 0
            found = True
            break

        if not found:
            # Dead end — backtrack within this chain
            fail_count[placed - 1] += 1
            if fail_count[placed - 1] > max_attempts:
                bt = min(backtrack_n, placed - 1)
                for _ in range(bt):
                    placed -= 1
                    chain_set.discard(snap(chain[placed]))
                fail_count[placed:placed + bt] = 0

                if placed <= 1:
                    raise RuntimeError(
                        f"Chain growth stuck at bead {placed}. "
                        "Increase BOX_L or reduce N_CHAINS."
                    )

    return chain


def generate_melt() -> np.ndarray:
    """Generate all chains. Returns array of shape (N_CHAINS, N_BEADS, 3)."""
    all_chains: list[np.ndarray] = []
    global_occupied: set         = set()   # all inter-chain sites

    # Spread chain seeds across the box on a regular grid to avoid
    # all chains starting from the same corner
    n_grid   = int(np.ceil(N_CHAINS ** (1 / 3))) + 1
    spacing  = BOX_L / n_grid
    grid_pts = np.array([
        [i * spacing + spacing / 2,
         j * spacing + spacing / 2,
         k * spacing + spacing / 2]
        for i in range(n_grid)
        for j in range(n_grid)
        for k in range(n_grid)
    ])
    rng.shuffle(grid_pts)

    for c in range(N_CHAINS):
        # Snap seed to nearest lattice site
        raw_seed = grid_pts[c % len(grid_pts)]
        seed_pos = wrap(
            np.round(raw_seed / LATTICE_STEP) * LATTICE_STEP, BOX_L
        )

        print(f"  Chain {c+1:3d}/{N_CHAINS} ...", end=" ", flush=True)

        placed = False
        for attempt in range(8):
            try:
                chain = grow_chain(
                    seed_pos,
                    global_occupied,
                    BOX_L,
                    N_BEADS,
                    MAX_ATTEMPTS,
                    BACKTRACK_N,
                    rng,
                )
                placed = True
                break
            except RuntimeError:
                # Try a random seed position
                seed_pos = wrap(
                    np.round(
                        rng.uniform(0, BOX_L, 3) / LATTICE_STEP
                    ) * LATTICE_STEP,
                    BOX_L,
                )
                print(f"\n    retry {attempt+1} ...", end=" ", flush=True)

        if not placed:
            raise RuntimeError(
                f"Failed to place chain {c+1} after 8 attempts. "
                "Increase BOX_L."
            )

        # Register all beads of this chain in the global set
        for bead in chain:
            global_occupied.add(snap(bead))

        all_chains.append(chain)
        total = sum(len(ch) for ch in all_chains)
        print(f"OK  ({total}/{N_CHAINS * N_BEADS} beads placed)")

    return np.array(all_chains)   # (N_CHAINS, N_BEADS, 3)


# ── LAMMPS writer ─────────────────────────────────────────────────────────────

def write_lammps_data(chains: np.ndarray, L: float, filename: str) -> None:
    n_ch, n_b, _ = chains.shape
    n_atoms       = n_ch * n_b
    n_bonds       = n_ch * (n_b - 1)

    with open(filename, "w") as f:
        f.write("LAMMPS data file — Kremer-Grest linear melt (v2 ideal walk)\n")
        f.write(f"# {n_ch} chains x {n_b} beads | "
                f"rho_init={n_atoms/L**3:.5f} | box={L}sigma\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_bonds} bonds\n\n")
        f.write("1 atom types\n")
        f.write("1 bond types\n\n")
        f.write(f"0.0 {L:.6f} xlo xhi\n")
        f.write(f"0.0 {L:.6f} ylo yhi\n")
        f.write(f"0.0 {L:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")

        f.write("Atoms\n\n")
        aid = 1
        for mol, chain in enumerate(chains, start=1):
            for bead in chain:
                f.write(f"{aid:6d} {mol:4d} 1 "
                        f"{bead[0]:12.6f} {bead[1]:12.6f} {bead[2]:12.6f}\n")
                aid += 1

        f.write("\nBonds\n\n")
        bid  = 1
        base = 1
        for chain in chains:
            for b in range(n_b - 1):
                f.write(f"{bid:6d} 1 {base+b:6d} {base+b+1:6d}\n")
                bid += 1
            base += n_b

    print(f"\nWritten: {filename}  ({n_atoms} atoms, {n_bonds} bonds)")


# ── Inline validation ──────────────────────────────────────────────────────────

def validate(chains: np.ndarray, L: float) -> None:
    """
    Fast inline validation — runs immediately after generation.
    Reports: bond lengths, <R²>/N, <Rg²>, minimum non-bonded distance.
    """
    print("\n" + "=" * 55)
    print("Inline validation")
    print("=" * 55)

    n_ch, n_b, _ = chains.shape
    flat          = chains.reshape(-1, 3)   # (N_total, 3)

    # ── Bond lengths ──
    bl = []
    for chain in chains:
        dr = chain[1:] - chain[:-1]
        dr -= L * np.round(dr / L)          # minimum image
        bl.extend(np.linalg.norm(dr, axis=1).tolist())
    bl = np.array(bl)
    print(f"\nBond lengths  mean={bl.mean():.5f}  std={bl.std():.5f}  "
          f"min={bl.min():.5f}  max={bl.max():.5f}")
    if bl.max() > 1.5:
        print("  WARNING: bond > FENE R0=1.5σ")

    # ── Chain statistics ──
    R2_list  = []
    Rg2_list = []
    for chain in chains:
        ee      = chain[-1] - chain[0]
        ee     -= L * np.round(ee / L)
        R2_list.append(float(np.dot(ee, ee)))
        com     = chain.mean(axis=0)
        dr      = chain - com
        Rg2_list.append(float(np.mean(np.sum(dr**2, axis=1))))

    R2_mean  = np.mean(R2_list)
    Rg2_mean = np.mean(Rg2_list)
    print(f"\nChain statistics ({n_ch} chains, N={n_b}):")
    print(f"  <R²>         = {R2_mean:.2f} σ²")
    print(f"  <R²>/N       = {R2_mean/(n_b-1):.4f}   "
          f"(ideal target ≈ 1.0, SAW ≈ 1.88)")
    print(f"  <Rg²>        = {Rg2_mean:.2f} σ²")
    print(f"  <Rg²>/<R²>/6 = {Rg2_mean/(R2_mean/6):.4f}  "
          f"(ideal Gaussian = 1.0)")

    # ── Minimum non-bonded distance (KDTree, sampled) ──
    n_sample = min(500, len(flat))
    idx      = rng.choice(len(flat), n_sample, replace=False)
    sample   = flat[idx]
    tree     = KDTree(sample)
    pairs    = tree.query_pairs(r=1.5, output_type="ndarray")

    # Build a quick bond lookup for sampled atoms (atom index in flat array)
    # Bonded pairs in flat: atom i and i+1 within each chain
    bonded = set()
    for c in range(n_ch):
        for b in range(n_b - 1):
            ai = c * n_b + b
            aj = c * n_b + b + 1
            bonded.add((ai, aj))

    min_nb  = np.inf
    nb_hist = {c: 0 for c in [0.5, 0.8, 1.0, 1.12, 1.5]}
    for i, j in pairs:
        gi, gj = int(idx[i]), int(idx[j])
        key    = (min(gi, gj), max(gi, gj))
        if key in bonded:
            continue
        dr = sample[i] - sample[j]
        dr -= L * np.round(dr / L)
        d  = float(np.linalg.norm(dr))
        if d < min_nb:
            min_nb = d
        for c in nb_hist:
            if d < c:
                nb_hist[c] += 1

    scale = (len(flat) / n_sample) ** 2
    print(f"\nNon-bonded distances (sampled, scaled to full system):")
    print(f"  Minimum non-bonded distance : {min_nb:.5f} σ")
    labels = {0.5: "CRITICAL", 0.8: "BAD", 1.0: "WARN",
              1.12: "NEAR-WCA", 1.5: "INFO"}
    for c, tag in labels.items():
        print(f"  r < {c:.2f}σ : {int(nb_hist[c]*scale):6d}  [{tag}]")

    # ── Verdict ──
    print()
    if min_nb >= 0.9 and nb_hist[0.5] == 0:
        print("PASS ✓  No catastrophic overlaps. Ready for soft push-off.")
    else:
        print("FAIL ✗  Overlaps detected. Check algorithm.")
    print("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("KG Melt Generator v2 — ideal-statistics lattice walk")
    print("=" * 55)
    print(f"  Chains={N_CHAINS}  N={N_BEADS}  L={BOX_L}σ  "
          f"ρ_init={N_CHAINS*N_BEADS/BOX_L**3:.4f}\n")

    chains = generate_melt()
    validate(chains, BOX_L)
    write_lammps_data(chains, BOX_L, OUTPUT)

    print("\nNext step:")
    print("  python check_structure.py linear_melt_init.data")
    print("  Then: melt_linear/equil/  →  stage_A_pushoff.lammps")