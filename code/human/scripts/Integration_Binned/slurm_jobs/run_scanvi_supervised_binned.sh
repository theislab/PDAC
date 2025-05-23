#!/bin/bash
#SBATCH --job-name=run_scanvi_supervised_binned
#SBATCH --get-user-env
#SBATCH --time=24:00:00
#SBATCH --nice=10000

#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=20

##SBATCH -p cpu_p
##SBATCH --qos=cpu_long
##SBATCH --cpus-per-task=20
##SBATCH --mem=400G

#SBATCH --output=out/run_scanvi_supervised_binned.out
#SBATCH --error=out/run_scanvi_supervised_binned.err

python3 -W ignore::DeprecationWarning /lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/scripts/run_scanvi_supervised_binned.py
