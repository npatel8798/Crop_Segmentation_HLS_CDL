#!/bin/bash
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=npatel23@umbc.edu   # Where to send mail
#SBATCH -D .
#SBATCH --job-name=testArray
#SBATCH --array=1-20
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --constraint=rtx_2080                  # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)
#SBATCH --cpus-per-task=4 

conda_env_name="/home/npatel23/gokhale_ada/users/npatel23/conda_envs/cropclassi"
conda activate $conda_env_name


states=("TENNESSEE" "TEXAS" "UTAH" "VERMONT" "VIRGINIA" "WASHINGTON" 
        "WEST_VIRGINIA" "WISCONSIN" "WYOMING")


state_name=${states[$SLURM_ARRAY_TASK_ID]}

echo $state_name

# Run the Python script with the state name as an argument
python generating_bounding_box.py "$state_name"