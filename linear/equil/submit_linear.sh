#!/bin/bash
#SBATCH --job-name=linear_equil
#SBATCH --partition=phd_student
#SBATCH --qos=phd_student
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

if [ "x$SLURM_JOB_ID" == "x" ]; then
    echo "Submit using sbatch"
    exit 1
fi

module purge
module load lammps-openmpi

cd $SLURM_SUBMIT_DIR

mkdir -p output

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Tasks  : $SLURM_NTASKS"

mpirun -np $SLURM_NTASKS lmp \
-in linear_chain.lammps \
-log output/lammps.log