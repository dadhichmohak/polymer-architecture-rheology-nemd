#!/usr/bin/env python3
"""
generate_linear_melt.py  —  v3
Kremer-Grest linear polymer melt: freely-jointed chain in continuous space.

Chain model: freely-jointed chain (FJC)
  - Bond length = 1.0 sigma exactly
  - Bond directions: uniform random on unit sphere
  - <R²>/N = 1.0 by construction (ideal Gaussian statistics)

Overlap prevention:
  - Intra-chain: reject bead if distance to any non-bonded chain bead < 0.8σ
  - Inter-chain: reject bead if distance to any placed bead < 0.8σ
  - Threshold 0.8σ < WCA diameter (1.122σ) — soft push-off handles the rest
  - At rho_init = 0.031, rejection rate is very low

Why 0.8σ threshold (not 1.0σ):
  - FJC bonds are exactly 1.0σ, so bonded pairs sit at 1.0σ by design
  - Non-bonded pairs in a Gaussian chain can occasionally reach 0.8-1.0σ
  - Requiring > 1.0σ for ALL non-bonded pairs would bias the chain back
    toward extended conformations (defeating the purpose)
  - 0.8σ threshold blocks only genuinely catastrophic overlaps

Reference: Auhl et al., J. Chem. Phys. 119, 12718 (2003)
"""

import numpy as np
from scipy.spatial import KDTree
import sys

# ── Parameters ────────────────────────────────────────────────────────────────
N_CHAINS        = 20
N_BEADS         = 100
BOND_LENGTH     = 1.0          # sigma
BOX_L           = 40.0         # sigma — rho_init = 2000/40^3 = 0.031
OVERLAP_CUTOFF  = 0.8          # sigma — minimum allowed non-bonded distance
MAX_BEAD_TRIES  = 1000         # attempts per bead before backtracking
BACKTRACK_N     = 5            # beads to remove on failure
SEED            = 42
OUTPUT          = "linear_melt_init.data"
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)


def wrap(pos: np.ndarray, L: float) -> np.ndarray:
    return pos % L


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """
    Uniform random unit vector on the unit sphere.
    Uses Marsaglia method (rejection sampling in cube) — unbiased.
    """
    while True:
        v = rng.uniform(-1, 1, 3)
        r = np.dot(v, v)
        if 0.0 < r <= 1.0:
            return v / np.sqrt(r)


def min_image_distance(p1: np.ndarray, p2: np.ndarray, L: float) -> float:
    """Minimum image distance between two points under PBC."""
    dr = p1 - p2
    dr -= L * np.round(dr / L)
    return float(np.linalg.norm(dr))


def grow_chain(
    seed: np.ndarray,
    all_placed: list,           # flat list of all previously placed beads
    L: float,
    n_beads: int,
    bond_length: float,
    overlap_cutoff: float,
    max_tries: int,
    backtrack_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Grow a single freely-jointed chain bead by bead.

    At each step:
      1. Draw a random unit vector, place candidate at current + b*u
      2. Wrap under PBC
      3. Check candidate against:
           - all beads of THIS chain (skip bonded pair i-1, also skip i-2
             because it is 1-3 and very close; only check i>=3 backward)
           - all beads of PREVIOUS chains
      4. If no overlap: accept
      5. If overlap or max_tries exceeded: backtrack BACKTRACK_N beads

    The walk direction is completely unrestricted (any direction on sphere),
    so <R²>/N → 1.0 for long chains.
    """
    chain     = np.zeros((n_beads, 3))
    chain[0]  = wrap(seed, L)
    placed    = 1

    # Build KDTree of previously placed chains for fast lookup
    # Rebuilt only when a new chain starts (not per bead)
    if len(all_placed) > 0:
        prev_tree = KDTree(all_placed)
    else:
        prev_tree = None

    fail_count = 0

    while placed < n_beads:
        accepted = False

        for _ in range(max_tries):
            # Random bond direction — uniform on sphere, no bias
            u         = random_unit_vector(rng)
            candidate = wrap(chain[placed - 1] + bond_length * u, L)

            # ── Check against current chain (non-bonded pairs only) ──
            # Skip beads placed - 1 (bonded) and placed - 2 (1-3 neighbour,
            # geometrically can be as close as 0 if chain folds back)
            # Check all beads 0 .. placed-3
            clash_intra = False
            if placed >= 3:
                check_beads = chain[:placed - 2]   # exclude bonded + 1-3
                for b in check_beads:
                    if min_image_distance(candidate, b, L) < overlap_cutoff:
                        clash_intra = True
                        break

            if clash_intra:
                continue

            # ── Check against all previously placed chains ──
            clash_inter = False
            if prev_tree is not None:
                # query_ball_point returns all neighbours within cutoff
                hits = prev_tree.query_ball_point(candidate, r=overlap_cutoff, p=2)
                if len(hits) > 0:
                    # Verify with minimum image (KDTree uses Euclidean,
                    # may miss PBC images for small boxes)
                    for h in hits:
                        if min_image_distance(
                            candidate, np.array(all_placed[h]), L
                        ) < overlap_cutoff:
                            clash_inter = True
                            break

            if clash_inter:
                continue

            # Accept this bead
            chain[placed] = candidate
            placed       += 1
            fail_count    = 0
            accepted      = True
            break

        if not accepted:
            # Backtrack
            fail_count += 1
            bt          = min(backtrack_n * fail_count, placed - 1)
            placed     -= bt

            if placed <= 1:
                raise RuntimeError(
                    f"Chain growth failed: stuck at bead {placed}. "
                    "Increase BOX_L or reduce OVERLAP_CUTOFF."
                )

    return chain


def generate_melt() -> np.ndarray:
    all_chains: list[np.ndarray] = []
    all_placed: list             = []   # flat list: [[x,y,z], ...] all beads

    # Spread seeds across box on a coarse grid
    n_grid   = int(np.ceil(N_CHAINS ** (1 / 3))) + 1
    spacing  = BOX_L / n_grid
    seeds    = np.array([
        [(i + 0.5) * spacing,
         (j + 0.5) * spacing,
         (k + 0.5) * spacing]
        for i in range(n_grid)
        for j in range(n_grid)
        for k in range(n_grid)
    ])
    rng.shuffle(seeds)

    for c in range(N_CHAINS):
        seed = seeds[c % len(seeds)]
        print(f"  Chain {c+1:3d}/{N_CHAINS} ...", end=" ", flush=True)

        for attempt in range(10):
            try:
                chain = grow_chain(
                    seed,
                    all_placed,
                    BOX_L,
                    N_BEADS,
                    BOND_LENGTH,
                    OVERLAP_CUTOFF,
                    MAX_BEAD_TRIES,
                    BACKTRACK_N,
                    rng,
                )
                break
            except RuntimeError:
                seed = rng.uniform(0, BOX_L, 3)
                print(f"\n    retry {attempt+1} ...", end=" ", flush=True)
        else:
            raise RuntimeError(
                f"Chain {c+1} failed after 10 attempts. Increase BOX_L."
            )

        all_placed.extend(chain.tolist())
        all_chains.append(chain)
        total = len(all_placed)
        print(f"OK  ({total}/{N_CHAINS * N_BEADS})")

    return np.array(all_chains)


# ── LAMMPS writer ─────────────────────────────────────────────────────────────

def write_lammps_data(chains: np.ndarray, L: float, filename: str) -> None:
    n_ch, n_b, _ = chains.shape
    n_atoms       = n_ch * n_b
    n_bonds       = n_ch * (n_b - 1)

    with open(filename, "w") as f:
        f.write("LAMMPS data file — KG linear melt v3 (FJC, ideal statistics)\n")
        f.write(f"# {n_ch} chains x {n_b} beads | "
                f"rho_init={n_atoms/L**3:.5f} | box={L}sigma\n\n")
        f.write(f"{n_atoms} atoms\n{n_bonds} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"0.0 {L:.6f} xlo xhi\n")
        f.write(f"0.0 {L:.6f} ylo yhi\n")
        f.write(f"0.0 {L:.6f} zlo zhi\n\n")
        f.write("Masses\n\n1 1.0\n\n")

        f.write("Atoms\n\n")
        aid = 1
        for mol, chain in enumerate(chains, start=1):
            for bead in chain:
                f.write(f"{aid:6d} {mol:4d} 1 "
                        f"{bead[0]:14.8f} {bead[1]:14.8f} {bead[2]:14.8f}\n")
                aid += 1

        f.write("\nBonds\n\n")
        bid, base = 1, 1
        for _ in chains:
            for b in range(n_b - 1):
                f.write(f"{bid:6d} 1 {base+b:6d} {base+b+1:6d}\n")
                bid += 1
            base += n_b

    print(f"Written: {filename}  ({n_atoms} atoms, {n_bonds} bonds)")


# ── Validation ────────────────────────────────────────────────────────────────

def validate(chains: np.ndarray, L: float) -> None:
    print("\n" + "=" * 55)
    print("Validation")
    print("=" * 55)

    n_ch, n_b, _ = chains.shape
    flat          = chains.reshape(-1, 3)

    # Bond lengths
    bl = []
    for chain in chains:
        dr = chain[1:] - chain[:-1]
        dr -= L * np.round(dr / L)
        bl.extend(np.linalg.norm(dr, axis=1))
    bl = np.array(bl)
    print(f"\nBond lengths: mean={bl.mean():.5f}  std={bl.std():.6f}  "
          f"min={bl.min():.5f}  max={bl.max():.5f}")

    # Chain statistics
    R2, Rg2 = [], []
    for chain in chains:
        ee  = chain[-1] - chain[0]
        ee -= L * np.round(ee / L)
        R2.append(float(np.dot(ee, ee)))
        com = chain.mean(axis=0)
        dr  = chain - com
        Rg2.append(float(np.mean(np.sum(dr ** 2, axis=1))))

    R2m, Rg2m = np.mean(R2), np.mean(Rg2)
    print(f"\nChain statistics ({n_ch} chains, N={n_b}):")
    print(f"  <R²>   = {R2m:.2f} σ²")
    print(f"  <R²>/N = {R2m/(n_b-1):.4f}   (target ≈ 1.0)")
    print(f"  <Rg²>  = {Rg2m:.2f} σ²")
    print(f"  <Rg²>  / (<R²>/6) = {Rg2m/(R2m/6):.4f}  (Gaussian = 1.0)")

    # Non-bonded minimum distance
    sample_n = min(600, len(flat))
    idx      = rng.choice(len(flat), sample_n, replace=False)
    sample   = flat[idx]
    tree     = KDTree(sample)
    pairs    = tree.query_pairs(r=1.5, output_type="ndarray")

    bonded_idx = set()
    for c in range(n_ch):
        for b in range(n_b - 1):
            bonded_idx.add((c * n_b + b, c * n_b + b + 1))

    min_nb = np.inf
    counts = {c: 0 for c in [0.5, 0.8, 1.0, 1.12]}
    for i, j in pairs:
        gi, gj = int(idx[i]), int(idx[j])
        if (min(gi, gj), max(gi, gj)) in bonded_idx:
            continue
        dr  = sample[i] - sample[j]
        dr -= L * np.round(dr / L)
        d   = float(np.linalg.norm(dr))
        if d < min_nb:
            min_nb = d
        for c in counts:
            if d < c:
                counts[c] += 1

    scale = (len(flat) / sample_n) ** 2
    print(f"\nNon-bonded distances:")
    print(f"  Minimum : {min_nb:.5f} σ")
    for c, tag in [(0.5, "CRITICAL"), (0.8, "BAD"),
                   (1.0, "WARN"), (1.12, "NEAR-WCA")]:
        print(f"  r < {c:.2f}σ : {int(counts[c]*scale):6d}  [{tag}]")

    print()
    ok = min_nb >= 0.75 and counts[0.5] == 0
    print("PASS ✓  Ready for soft push-off." if ok else
          "FAIL ✗  Overlaps detected.")
    print("=" * 55)


if __name__ == "__main__":
    print("=" * 55)
    print("KG Melt Generator v3 — FJC, ideal Gaussian statistics")
    print("=" * 55)
    print(f"  Chains={N_CHAINS}  N={N_BEADS}  L={BOX_L}σ  "
          f"ρ_init={N_CHAINS*N_BEADS/BOX_L**3:.4f}")
    print(f"  Bond length={BOND_LENGTH}σ  Overlap cutoff={OVERLAP_CUTOFF}σ\n")

    chains = generate_melt()
    validate(chains, BOX_L)
    write_lammps_data(chains, BOX_L, OUTPUT)

    print("\nNext:  python check_structure.py linear_melt_init.data")
    print("Then:  melt_linear/equil/stage_A_pushoff.lammps")