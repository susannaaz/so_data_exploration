#!/bin/bash -l

#SBATCH --account=simonsobs
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=SATp3-iso-stats
#SBATCH --mail-user sazzoni@princeton.edu
#SBATCH --mail-type all

set -e

module use --append /scratch/gpfs/SIMONSOBS/modules
module load soconda/3.12/v0.6.8

export SOTODLIB_RESOURCES='{"de421.bsp": "file:///scratch/gpfs/SIMONSOBS/planets/de421.bsp"}'

# Log file
log="/scratch/gpfs/SIMONSOBS/users/sa5705/SO/ISO_v3/iso_stats/log_iso_stats_satp3_20251208.log"

#=========== Compute runtime parameters ============

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=2

launch_str="srun -n 224 --cpus-per-task=2 --export=ALL python"

#=========== Run it ============


com="${launch_str} /home/sa5705/software/so_data_exploration/tiger/scratch/iso_v3_stats_satp3_20251208.py \
/home/sa5705/software/iso-sat/v3/preprocessing/satp3/preprocessing_config_20250801_init.yaml \
/home/sa5705/software/iso-sat/v3/preprocessing/satp3/preprocessing_config_20250801_proc.yaml \
--nproc 224 \
--savename /scratch/gpfs/SIMONSOBS/users/sa5705/SO/ISO_v3/iso_stats/20251208_cut_stats_satp3.h5 \
--errlog-ext ${log}"

#--noise-range 2e-6 80e-6 \

echo ${com}
echo "Launching pipeline at $(date)"
eval ${com} > ${log} 2>&1

echo "Ending batch script at $(date)"
