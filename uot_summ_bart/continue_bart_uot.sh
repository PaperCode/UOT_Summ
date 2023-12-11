#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=450000M
#SBATCH --time=UNLIMITED
#SBATCH --partition=A6000
#SBATCH --nodelist=seis19
hostname
allennlp train continue_bart_uot.jsonnet -s model --include-package model_classes --recover