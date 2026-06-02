# Polymer Architecture Classification via NEMD

A computational rheology project focused on identifying polymer architectures using Non-Equilibrium Molecular Dynamics (NEMD) simulations in LAMMPS.

## Project Overview

Bio-derived polymers often exhibit complex molecular architectures that significantly influence their rheological response under flow.

This project builds a simulation library of different polymer architectures and characterizes them through:

* Shear viscosity (η)
* First Normal Stress Difference (N1)
* Power-law rheology
* Orientation tensor analysis

The ultimate objective is to classify an unknown polymer architecture from its rheological signature.

---

## Architectures

### Linear Polymer

* Kremer-Grest bead-spring model
* Single-chain validation
* Melt simulations

### Star Polymer

* Multi-arm branched architecture
* Single-chain validation
* Melt simulations

### H-Shaped Polymer

* Planned implementation

---

## Simulation Workflow

### 1. Structure Generation

Generate coarse-grained polymer configurations:

* Linear chains
* Star polymers
* H-shaped polymers

---

### 2. Equilibration

#### Stage A

Soft push-off

* Harmonic bonds
* Soft pair potential
* NVE/limit dynamics

#### Stage B

Gradual compression

Target density:

ρ = 0.85

#### Stage C

Kremer-Grest equilibration

* FENE bonds
* WCA interactions

#### Stage D

Extended equilibration

Production-quality melt configurations

---

### 3. NEMD

Shear flow imposed using:

* SLLOD algorithm
* Triclinic simulation box
* Lees-Edwards style deformation

Shear rates:

* 0.001
* 0.003
* 0.01
* 0.03
* 0.1

---

## Rheological Quantities

### Viscosity

η = -Pxy / γ̇

### First Normal Stress Difference

N1 = Pxx − Pyy

---

## Power-Law Model

τ = K γ̇ⁿ

Extracted parameters:

* Flow consistency index K
* Flow behavior index n

---

## Orientation Tensor

Sij = 1/2 < 3uiuj − δij >

Used for polymer architecture identification.

---

## Repository Structure

```text
linear/
├── equil/
├── nemd/
└── results/

star/
├── equil/
├── nemd/
└── results/

melt_linear/
├── build/
├── equil/
├── nemd/
└── results/

melt_star/
├── build/
├── equil/
├── nemd/
└── results/

h_shaped/
└── (planned)
```

---

## Software

* LAMMPS
* Python
* NumPy
* SciPy
* Matplotlib

---

## Results

Example (Linear Melt):

| γ̇    | η     |
| ----- | ----- |
| 0.001 | 22.07 |
| 0.003 | 22.01 |
| 0.01  | 11.60 |
| 0.03  | 6.52  |
| 0.10  | 3.62  |

Observed behavior:

* Strong shear thinning
* Non-Newtonian rheology

Power-law fit:

K = 1.513

n = 0.579

---

## Future Work

* Complete H-shaped architecture library
* Unknown polymer classification
* Automated rheology database generation
* Machine-learning-based architecture prediction

---

## Author

Mohak Dadhich

Chemical Engineering

Indian Institute of Technology Indore

---

## License

MIT License
