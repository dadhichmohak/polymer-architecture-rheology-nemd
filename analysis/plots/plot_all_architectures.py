import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
DATA_DIR = ANALYSIS_DIR / "data"
PLOTS_DIR = ANALYSIS_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load rheology data
# --------------------------------------------------

lin = np.loadtxt(DATA_DIR / "rheology_linear.dat", skiprows=1)
star = np.loadtxt(DATA_DIR / "rheology_star.dat", skiprows=1)
h = np.loadtxt(DATA_DIR / "rheology_h.dat", skiprows=1)

# --------------------------------------------------
# Viscosity vs shear rate
# --------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(lin[:,0], lin[:,1], 'o-', label='Linear')
plt.plot(star[:,0], star[:,1], 's-', label='Star')
plt.plot(h[:,0], h[:,1], '^-', label='H-shaped')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Shear rate $\dot{\gamma}$')
plt.ylabel(r'Viscosity $\eta$')
plt.title('Viscosity vs Shear Rate')

plt.legend()
plt.tight_layout()

plt.savefig(PLOTS_DIR / "viscosity_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# N1 vs shear rate
# --------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(lin[:,0], np.abs(lin[:,2]), 'o-', label='Linear')
plt.plot(star[:,0], np.abs(star[:,2]), 's-', label='Star')
plt.plot(h[:,0], np.abs(h[:,2]), '^-', label='H-shaped')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Shear rate $\dot{\gamma}$')
plt.ylabel(r'$|N_1|$')
plt.title('First Normal Stress Difference')

plt.legend()
plt.tight_layout()

plt.savefig(PLOTS_DIR / "N1_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# Power-law comparison
# --------------------------------------------------

architectures = ["Linear", "Star", "H-shaped"]

K = [
    1.512947,
    1.422605,
    1.566561
]

n = [
    0.579453,
    0.520028,
    0.575288
]

plt.figure(figsize=(6,4))
plt.bar(architectures, K)

plt.ylabel("K")
plt.title("Consistency Index")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "K_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(6,4))
plt.bar(architectures, n)

plt.ylabel("n")
plt.title("Power-law Index")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "n_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# Orientation comparison
# --------------------------------------------------

lambda_max = [
    0.469606,
    0.386658,
    0.393195
]

plt.figure(figsize=(6,4))
plt.bar(architectures, lambda_max)

plt.ylabel(r'$\lambda_{max}$')
plt.title('Maximum Orientation Eigenvalue')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "orientation_comparison.png", dpi=300)
plt.close()

print("\nGenerated:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(" ", f.name)