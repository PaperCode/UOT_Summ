#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30000M
#SBATCH --time=UNLIMITED
#SBATCH --partition=Titan
#SBATCH --nodelist=seis18
hostname
python -u main.py --mode continue --pattern ot --cuda_id 2 -e cnndm.s2s.lstm.gpu2.epoch127.4 | tee train.log
