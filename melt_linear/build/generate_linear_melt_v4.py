#!/usr/bin/env python3
"""
generate_linear_melt.py  —  v4
KG linear melt: freely-jointed chain, near-ideal Gaussian statistics.

Key fixes vs v3:
  - Intra-chain exclusion threshold lowered to 0.5σ (was 0.8σ)
    → removes SAW bias while still blocking catastrophic self-overlaps
  - Inter-chain exclusion threshold kept at 0.8σ
  - Rg² validation corrected for PBC unwrapping
"""

import numpy as np
from scipy.spatial import KDTree
import sys

# ── Parameters ────────────────────────────────────────────────────────────────
N_CHAINS            = 20
N_BEADS             = 100
BOND_LENGTH         = 1.0
BOX_L               = 40.0
INTRA_CUTOFF        = 0.5     # sigma — intra-chain catastrophic overlap only
INTER_CUTOFF        = 0.8     # sigma — inter-chain overlap threshold
MAX_BEAD_TRIES      = 2000
BACKTRACK_N         = 5
SEED                = 42
OUTPUT              = "linear_melt_init.data"
# ──────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)


def wrap(pos: np.ndarray, L: float) -> np.ndarray:
    return pos % L


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit vector — Marsaglia method, no bias."""
    while True:
        v = rng.uniform(-1.0, 1.0, 3)
        r2 = np.dot(v, v)
        if 0.0 < r2 <= 1.0:
            return v / np.sqrt(r2)


def min_image_dist(p1: np.ndarray, p2: np.ndarray, L: float) -> float:
    dr = p1 - p2
    dr -= L * np.round(dr / L)
    return float(np.linalg.norm(dr))


def unwrap_chain(chain: np.ndarray, L: float) -> np.ndarray:
    """
    Unwrap a PBC-wrapped chain into continuous space.
    Each bond vector is corrected to its minimum image.
    Required for correct Rg² calculation.
    """
    unwrapped = chain.copy()
    for i in range(1, len(chain)):
        dr = chain[i] - unwrapped[i - 1]
        dr -= L * np.round(dr / L)          # minimum image bond vector
        unwrapped[i] = unwrapped[i - 1] + dr
    return unwrapped


def grow_chain(
    seed: np.ndarray,
    all_placed: list,
    L: float,
    n_beads: int,
    bond_length: float,
    intra_cutoff: float,
    inter_cutoff: float,
    max_tries: int,
    backtrack_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Grow one FJC chain bead by bead.

    Exclusion rules:
      Intra-chain: reject candidate if distance to any bead
                   placed-3 or earlier < intra_cutoff (0.5σ)
                   → blocks only catastrophic self-overlaps
                   → does NOT create SAW bias (threshold too low
                      to influence Gaussian statistics)

      Inter-chain: reject candidate if distance to any bead
                   in previously placed chains < inter_cutoff (0.8σ)

    Bond direction: uniform random on unit sphere — no directional bias.
    """
    chain    = np.zeros((n_beads, 3))
    chain[0] = wrap(seed, L)
    placed   = 1

    prev_tree = KDTree(all_placed) if len(all_placed) > 0 else None

    fail_streak = 0

    while placed < n_beads:
        accepted = False

        for _ in range(max_tries):
            u         = random_unit_vector(rng)
            candidate = wrap(chain[placed - 1] + bond_length * u, L)

            # ── Intra-chain check (catastrophic only, threshold 0.5σ) ──
            # Skip beads placed-1 (bonded) and placed-2 (1-3 neighbour)
            # Only check beads 0 .. placed-3
            clash = False
            if placed >= 3:
                for k in range(placed - 2):
                    if min_image_dist(candidate, chain[k], L) < intra_cutoff:
                        clash = True
                        break
            if clash:
                continue

            # ── Inter-chain check (threshold 0.8σ) ──
            if prev_tree is not None:
                hits = prev_tree.query_ball_point(
                    candidate, r=inter_cutoff, p=2
                )
                for h in hits:
                    if min_image_dist(
                        candidate, np.array(all_placed[h]), L
                    ) < inter_cutoff:
                        clash = True
                        break
            if clash:
                continue

            # Accept
            chain[placed] = candidate
            placed       += 1
            fail_streak   = 0
            accepted      = True
            break

        if not accepted:
            fail_streak += 1
            bt = min(backtrack_n * fail_streak, placed - 1)
            placed -= bt
            if placed <= 1:
                raise RuntimeError(
                    f"Stuck at bead {placed}. "
                    "Increase BOX_L or lower INTRA_CUTOFF."
                )

    return chain


def generate_melt() -> np.ndarray:
    all_chains: list[np.ndarray] = []
    all_placed: list             = []

    n_grid  = int(np.ceil(N_CHAINS ** (1 / 3))) + 1
    spacing = BOX_L / n_grid
    seeds   = np.array([
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
                    seed, all_placed, BOX_L, N_BEADS,
                    BOND_LENGTH, INTRA_CUTOFF, INTER_CUTOFF,
                    MAX_BEAD_TRIES, BACKTRACK_N, rng,
                )
                break
            except RuntimeError:
                seed = rng.uniform(0, BOX_L, 3)
                print(f"\n    retry {attempt+1}...", end=" ", flush=True)
        else:
            raise RuntimeError(
                f"Chain {c+1} failed after 10 attempts. Increase BOX_L."
            )

        all_placed.extend(chain.tolist())
        all_chains.append(chain)
        print(f"OK  ({len(all_placed)}/{N_CHAINS * N_BEADS})")

    return np.array(all_chains)


# ── LAMMPS writer ─────────────────────────────────────────────────────────────

def write_lammps_data(chains: np.ndarray, L: float, filename: str) -> None:
    n_ch, n_b, _ = chains.shape
    n_atoms       = n_ch * n_b
    n_bonds       = n_ch * (n_b - 1)

    with open(filename, "w") as f:
        f.write("LAMMPS data file — KG linear melt v4\n")
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
    print(f"\nBond lengths : mean={bl.mean():.5f}  std={bl.std():.6f}  "
          f"min={bl.min():.5f}  max={bl.max():.5f}")

    # Chain statistics — using UNWRAPPED coordinates for Rg²
    R2_list, Rg2_list = [], []
    for chain in chains:
        # R² via minimum image end-to-end
        ee  = chain[-1] - chain[0]
        ee -= L * np.round(ee / L)
        R2_list.append(float(np.dot(ee, ee)))

        # Rg² via unwrapped chain (PBC-corrected)
        uw  = unwrap_chain(chain, L)
        com = uw.mean(axis=0)
        dr  = uw - com
        Rg2_list.append(float(np.mean(np.sum(dr ** 2, axis=1))))

    R2m  = np.mean(R2_list)
    Rg2m = np.mean(Rg2_list)
    print(f"\nChain statistics ({n_ch} chains, N={n_b}):")
    print(f"  <R²>            = {R2m:.2f} σ²")
    print(f"  <R²>/N          = {R2m / (n_b - 1):.4f}   (target ≈ 1.0)")
    print(f"  <Rg²>           = {Rg2m:.2f} σ²")
    print(f"  <Rg²> / (<R²>/6)= {Rg2m / (R2m / 6):.4f}  (Gaussian = 1.0)")

    # Non-bonded minimum distance
    sample_n = min(600, len(flat))
    idx      = rng.choice(len(flat), sample_n, replace=False)
    sample   = flat[idx]
    tree     = KDTree(sample)
    pairs    = tree.query_pairs(r=1.5, output_type="ndarray")

    bonded_global = set()
    for c in range(n_ch):
        for b in range(n_b - 1):
            ai = c * n_b + b
            aj = c * n_b + b + 1
            bonded_global.add((ai, aj))

    min_nb = np.inf
    counts = {c: 0 for c in [0.5, 0.8, 1.0, 1.12]}
    for i, j in pairs:
        gi, gj = int(idx[i]), int(idx[j])
        key    = (min(gi, gj), max(gi, gj))
        if key in bonded_global:
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
    print(f"\nNon-bonded distances (sampled, scaled):")
    print(f"  Minimum : {min_nb:.5f} σ")
    for c, tag in [(0.5, "CRITICAL"), (0.8, "BAD"),
                   (1.0, "WARN"),     (1.12, "NEAR-WCA")]:
        print(f"  r < {c:.2f}σ : {int(counts[c] * scale):6d}  [{tag}]")

    print()
    ok = (min_nb >= 0.45 and counts[0.5] == 0)
    print("PASS ✓  Ready for soft push-off." if ok
          else "FAIL ✗  Overlaps detected.")
    print("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("KG Melt Generator v4 — FJC, near-ideal statistics")
    print("=" * 55)
    print(f"  N_CHAINS={N_CHAINS}  N_BEADS={N_BEADS}  "
          f"L={BOX_L}σ  ρ={N_CHAINS*N_BEADS/BOX_L**3:.4f}")
    print(f"  intra_cutoff={INTRA_CUTOFF}σ  "
          f"inter_cutoff={INTER_CUTOFF}σ\n")

    chains = generate_melt()
    validate(chains, BOX_L)
    write_lammps_data(chains, BOX_L, OUTPUT)

    print("\nNext:  python check_structure.py linear_melt_init.data")
    print("Then:  melt_linear/equil/stage_A_pushoff.lammps")