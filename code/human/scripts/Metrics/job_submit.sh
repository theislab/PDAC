#!/bin/bash
#SBATCH --job-name=metrics_scib
#SBATCH --get-user-env
#SBATCH --time=48:00:00
#SBATCH --nice=10000

##SBATCH -p gpu_p
##SBATCH --qos=gpu_long
##SBATCH --gres=gpu:2 
##SBATCH --mem=200G
##SBATCH --cpus-per-task=20

#SBATCH -p cpu_p
#SBATCH --qos=cpu_long
#SBATCH --cpus-per-task=20
#SBATCH --mem=400G

#SBATCH --output=metrics_scib.out
#SBATCH --error=metrics_scib.err

python3 -W ignore::DeprecationWarning /lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/scripts/metrics_scib.py