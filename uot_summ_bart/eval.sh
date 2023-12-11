#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=300000M
#SBATCH --time=UNLIMITED
#SBATCH --partition=A6000
#SBATCH --nodelist=seis19
hostname
allennlp evaluate model_12_epoch_epsilon0.05_tau0.3/model.tar.gz /gds/xshen/projdata17/researchPJ/processed_gov/test \
    --cuda-device 0 --include-package model_classes --output-file tmp18
