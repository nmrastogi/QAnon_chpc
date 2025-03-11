#!/bin/bash
#SBATCH --account soc-gpu-kp
#SBATCH --partition soc-gpu-kp
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --mail-user=naman.rastogi@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o slurmjob_twitter_rest_2-%j.out-%N
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_env

OUT_DIR=/uufs/chpc.utah.edu/common/home/u1472278/QAnon/results/
mkdir -p ${OUT_DIR}
python twitter.py --output_dir ${OUT_DIRd}