#!/bin/bash

#SBATCH --job-name=busi_mlrseg
#SBATCH --array=0-7
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_6000
#SBATCH --output=outfiles/final/pets/output_%A_%a.log
#SBATCH --error=outfiles/final/pets/output_%A_%a.err

DT=`date +"%m-%d_%H-%M"`

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

source activate unet

# cp /nfs/ada/oates/users/omkark1/Thesis_Work/Datasets.zip /scratch/$SLURM_JOBID
# unzip -q /scratch/$SLURM_JOBID/Datasets.zip -d /scratch/$SLURM_JOBID

# th=$(echo "0.04 * $SLURM_ARRAY_TASK_ID" | bc)

# echo "The threshold is: $th"

# if [[ "$SLURM_ARRAY_TASK_ID" -eq 6 ]]; then
# 	th=50
# elif [[ "$SLURM_ARRAY_TASK_ID" -eq 7 ]]; then
# 	th=-1
# else
# 	echo "Nothin"
# fi

th=(0 0.01 0.02 0.04 0.08 0.12 50 50)
md=('perc_loss_only' 'default' 'default' 'default' 'default' 'default' 'default' 'weak_mask_only')
echo "The threshold is: ${th[1]}"
echo "The threshold is: ${th[$SLURM_ARRAY_TASK_ID]}"
echo "The mode is: ${md[$SLURM_ARRAY_TASK_ID]}"
concatenated_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Concatenated ID: $concatenated_id"

python /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/pets_optuna_train_multiloss.py -pl=True -e=80 -b=40 -ir=224 -th=${th[$SLURM_ARRAY_TASK_ID]} -j=$concatenated_id -m=${md[$SLURM_ARRAY_TASK_ID]}
# mv output_$SLURM_JOBID.log /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.log
# mv output_$SLURM_JOBID.err /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.err

# mv output_$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID.log /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.log
# mv output_$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID.err /nfs/ada/oates/users/omkark1/Thesis_Work/UNetSeg_PyTorch/outfiles/busi/optuna/output_$DT.err