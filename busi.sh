#!/bin/bash

#SBATCH --job-name=busi_mlrseg
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_2080
#SBATCH --output=output_%j.log
#SBATCH --error=output_%j.err

DT=`date +"%m-%d_%H-%M"`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

source activate unet

# cp /nfs/ada/oates/users/omkark1/Thesis_Work/Datasets.zip /scratch/$SLURM_JOBID
# unzip -q /scratch/$SLURM_JOBID/Datasets.zip -d /scratch/$SLURM_JOBID

python /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/busi_optuna_train_multiloss.py -pl=True -e=160 -b=36 -ir=224 -th=50

mv output_$SLURM_JOBID.log /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.log
mv output_$SLURM_JOBID.err /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.err