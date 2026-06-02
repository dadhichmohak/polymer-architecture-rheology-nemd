#!/bin/bash
#SBATCH --job-name=nemd_03
#SBATCH --partition=phd_student
#SBATCH --qos=phd_student
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

module purge
module load lammps-openmpi

cd $SLURM_SUBMIT_DIR

mkdir -p output

mpirun -np $SLURM_NTASKS lmp \
-in linear_nemd.lammps \
-log output/nemd.log