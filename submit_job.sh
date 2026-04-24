#!/bin/bash
#SBATCH --job-name=My_Job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
source /soft/intel/parallel_studio_xe_2016.3.067/bin/psxevars.sh intel64
mpirun ./matrix_mpi
