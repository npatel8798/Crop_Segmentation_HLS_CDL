#!/bin/bash
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=npatel23@umbc.edu   # Where to send mail
#SBATCH -D .
#SBATCH --job-name=nebraska
#SBATCH --array=0-9
#SBATCH --output=nebraska_%A_%a.out
#SBATCH --error=nebraska_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --time=72:00:00
#SBATCH --constraint=rtx_2080                  # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)
#SBATCH --cpus-per-task=4 

conda_env_name="/home/npatel23/gokhale_ada/users/npatel23/conda_envs/cropclassi"
conda activate $conda_env_name


year=("2014" "2015" "2016" "2017" "2018" "2019" 
        "2020" "2021" "2022" "2023")


year_name=${year[$SLURM_ARRAY_TASK_ID]}

echo $year_name

# Run the Python script with the state name as an argument
python Data_Download.py nebraska "$year_name"