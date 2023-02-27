#!/bin/bash

#SBATCH --job-name=imrecon
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --output=/nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/imrecon_run_1.out
#SBATCH --error=/nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/imrecon_run_1.err


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

source activate unet

python /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/train_recon.py -b=96 -e=100
