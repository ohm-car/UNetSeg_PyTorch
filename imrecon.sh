#!/bin/bash

#SBATCH --job-name=mlrseg
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --output=output_%j.log
#SBATCH --error=output_%j.err

DT=`date +"%m-%d_%H-%M"`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

source activate unet

cp /nfs/ada/oates/users/omkark1/Thesis_Work/Datasets.zip /scratch/$SLURM_JOBID
unzip -q /scratch/$SLURM_JOBID/Datasets.zip -d /scratch/$SLURM_JOBID

# python /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/train_multiloss.py -rd=/scratch/$SLURM_JOBID -b=160 -e=60 -sf=10 -rw=0.1 -r='omkar'
python /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/optuna_train_multiloss.py -rd=/scratch/$SLURM_JOBID -b=128 -e=60

mv output_$SLURM_JOBID.log /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/multiloss/output_$DT.log
mv output_$SLURM_JOBID.err /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/multiloss/output_$DT.err