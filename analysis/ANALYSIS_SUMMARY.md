# Rheology Analysis Summary

## Architectures Completed

### Linear Melt

Power-law fit:

* K = 1.512947
* n = 0.579453

### Star Melt

Power-law fit:

* K = 1.422605
* n = 0.520028

---

## Viscosity Comparison

| Shear Rate |  Linear |    Star |
| ---------- | ------: | ------: |
| 0.001      | 22.0715 | 38.3464 |
| 0.003      | 22.0054 | 23.2764 |
| 0.01       | 11.5954 | 13.3515 |
| 0.03       |  6.5242 |  7.7106 |
| 0.1        |  3.6196 |  4.2058 |

Observation:

* Star melt exhibits higher viscosity across the investigated shear-rate range.

---

## First Normal Stress Difference

| Shear Rate | N1 Linear | N1 Star |
| ---------- | --------: | ------: |
| 0.001      |   -0.0767 | -0.0348 |
| 0.003      |   -0.2243 | -0.1366 |
| 0.01       |   -0.6305 | -0.4298 |
| 0.03       |   -1.4779 | -1.0365 |
| 0.1        |   -2.9512 | -2.3185 |

Observation:

* Magnitude of N1 increases with shear rate for both architectures.

---

## Power-Law Model

η = K γ̇^(n−1)

| Architecture |        K |        n |
| ------------ | -------: | -------: |
| Linear Melt  | 1.512947 | 0.579453 |
| Star Melt    | 1.422605 | 0.520028 |

Observation:

* Star melt exhibits stronger shear-thinning behavior (lower n).

---

## Generated Figures

* 01_viscosity_vs_shear_rate.png
* 02_N1_vs_shear_rate.png
* 03_publication_summary.png

---

## Next Steps

1. Implement H-shaped polymer architecture.
2. Perform equilibration and NEMD simulations.
3. Extend rheology comparison to all architectures.
4. Build predictive model relating architecture and rheology.
