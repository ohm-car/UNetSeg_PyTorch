#!/bin/bash

#SBATCH --job-name=test
#SBATCH --mail-user=omkark1@umbc.edu
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=10-12:00:00
#SBATCH --constraint=rtx_2080
#SBATCH --output=test.out
#SBATCH --error=test.err

python hello.py
