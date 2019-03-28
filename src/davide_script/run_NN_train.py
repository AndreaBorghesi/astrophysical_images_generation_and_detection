#!/bin/sh
#SBATCH --gres=gpu:tesla:4         # GPUs requested
#SBATCH --partition=dvd_usr_prod   # partition selected
#SBATCH --account=cin_powerdam_5    # account selected
#SBATCH -N 1      # nodes requested
#SBATCH
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=100000  # memory in Mb
# --open-mode=append
# --open-mode=truncate
#SBATCH -o logs/outfile_mse_input_4gpu_bn_996imgsize_32bs_100zdim
#SBATCH -e logs/errfile_mse_input_4gpu_bn_996imgsize_32bs_100zdim
#SBATCH -t 03:00:00  # time requested in hour:minute:second

#python train_ae_davide_withImgGen.py
python train_vae_davide_withImgGen.py
