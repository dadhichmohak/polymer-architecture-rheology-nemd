#!/bin/bash
#SBATCH --job-name=stageA_pushoff
#SBATCH --partition=compute
#SBATCH --qos=medium
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

if [ "x$SLURM_JOB_ID" == "x" ]; then
    echo "Submit with sbatch"
    exit 1
fi

module load lammps-openmpi
mkdir -p output

date_start=`date +"%H:%M:%S on %d %b %Y"`
echo "========================================="
echo "Stage A: Soft push-off"
echo "Started : $date_start"
echo "Job     : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "========================================="

cd $SLURM_SUBMIT_DIR
mpirun -np 4 lmp -in stage_A_pushoff.lammps -log output/stage_A.log

date_end=`date +"%H:%M:%S on %d %b %Y"`
echo "========================================="
echo "Finished: $date_end"
echo "========================================="