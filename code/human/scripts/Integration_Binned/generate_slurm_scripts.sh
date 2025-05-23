#!/bin/bash

# Directory containing Python scripts
SCRIPT_DIR="/lustre/groups/ml01/workspace/shrey.parikh/PDAC_Work_Dir/PDAC_Final/Binned_Data/scripts"

# Directory where SLURM job scripts will be saved
JOB_SCRIPT_DIR="$SCRIPT_DIR/slurm_jobs"

# Ensure the job script directory exists
mkdir -p $JOB_SCRIPT_DIR

# Loop over each Python script in the directory
for script in $SCRIPT_DIR/*.py; do
    # Extract the script name without the path and extension
    SCRIPT_NAME=$(basename "$script" .py)

    # Define the job script filename
    JOB_SCRIPT="$JOB_SCRIPT_DIR/${SCRIPT_NAME}.sh"

    # Create the job script
    cat <<EOL > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=$SCRIPT_NAME
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

#SBATCH --output=out/${SCRIPT_NAME}.out
#SBATCH --error=out/${SCRIPT_NAME}.err

python3 -W ignore::DeprecationWarning $SCRIPT_DIR/$SCRIPT_NAME.py
EOL

    # Make the script executable
    chmod +x $JOB_SCRIPT

    echo "Generated: $JOB_SCRIPT"
done