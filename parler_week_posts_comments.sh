#!/bin/bash
#SBATCH --account soc-gpu-kp
#SBATCH --partition soc-gpu-kp
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=2
#SBATCH --gres=gpu
#SBATCH --time=10:00:00
#SBATCH --mem=128GB
#SBATCH --mail-user=naman.rastogi@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o parler_post_comments-%j.out-%N
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_env

OUT_DIR=/uufs/chpc.utah.edu/common/home/u1472278/QAnon/results/
mkdir -p ${OUT_DIR}
python parler_week_post_comments.py --output_dir ${OUT_DIRd}