import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Data
# --------------------------------------------------

gdot = np.array([0.001, 0.003, 0.01, 0.03, 0.1])

eta_linear = np.array([
    22.0715,
    22.0054,
    11.5954,
    6.52423,
    3.61959
])

eta_star = np.array([
    38.346392,
    23.276373,
    13.351538,
    7.710630,
    4.205754
])

N1_linear = np.array([
    -0.0766664,
    -0.22432,
    -0.630471,
    -1.47788,
    -2.95120
])

N1_star = np.array([
    -0.034846,
    -0.136584,
    -0.429792,
    -1.036505,
    -2.318499
])

# --------------------------------------------------
# Figure 1
# Viscosity
# --------------------------------------------------

plt.figure(figsize=(7,5))

plt.plot(gdot, eta_linear,
         marker='o',
         linewidth=2,
         label='Linear Melt')

plt.plot(gdot, eta_star,
         marker='s',
         linewidth=2,
         label='Star Melt')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Shear Rate $\dot{\gamma}$')
plt.ylabel(r'Viscosity $\eta$')

plt.title('Viscosity vs Shear Rate')
plt.grid(True, which='both', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('01_viscosity_vs_shear_rate.png', dpi=600)
plt.close()

# --------------------------------------------------
# Figure 2
# N1
# --------------------------------------------------

plt.figure(figsize=(7,5))

plt.plot(gdot, np.abs(N1_linear),
         marker='o',
         linewidth=2,
         label='Linear Melt')

plt.plot(gdot, np.abs(N1_star),
         marker='s',
         linewidth=2,
         label='Star Melt')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Shear Rate $\dot{\gamma}$')
plt.ylabel(r'$|N_1|$')

plt.title('First Normal Stress Difference')
plt.grid(True, which='both', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('02_N1_vs_shear_rate.png', dpi=600)
plt.close()

# --------------------------------------------------
# Figure 3
# Combined publication figure
# --------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].plot(gdot, eta_linear, 'o-', linewidth=2,
           label='Linear Melt')

ax[0].plot(gdot, eta_star, 's-', linewidth=2,
           label='Star Melt')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$\dot{\gamma}$')
ax[0].set_ylabel(r'$\eta$')
ax[0].set_title('Viscosity')
ax[0].grid(True, which='both', alpha=0.3)
ax[0].legend()

ax[1].plot(gdot, np.abs(N1_linear), 'o-', linewidth=2,
           label='Linear Melt')

ax[1].plot(gdot, np.abs(N1_star), 's-', linewidth=2,
           label='Star Melt')

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel(r'$\dot{\gamma}$')
ax[1].set_ylabel(r'$|N_1|$')
ax[1].set_title('Normal Stress Difference')
ax[1].grid(True, which='both', alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig('03_publication_summary.png', dpi=600)
plt.close()

print("Plots written successfully.")