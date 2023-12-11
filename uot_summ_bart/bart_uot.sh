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
allennlp train bart_uot.jsonnet -s model_12_epoch_epsilon0.05_tau0.3 --include-package model_classes