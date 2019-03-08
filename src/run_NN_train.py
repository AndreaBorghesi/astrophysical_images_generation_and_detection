#!/bin/sh
#SBATCH --gres=gpu:tesla:1         # GPUs requested
#SBATCH --partition=dvd_usr_prod   # partition selected
#SBATCH --account=cin_powerdam_5    # account selected
#SBATCH -N 1      # nodes requested
#SBATCH
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=100000  # memory in Mb
# --open-mode=append
# --open-mode=truncate
#SBATCH -o outfile_mae_input_1gpu # send stdout to outfile
#SBATCH -e errfile_mae_input_1gpu  # send stderr to errfile
#SBATCH -t 03:00:00  # time requested in hour:minute:second

python train_ae_davide_withImgGen.py
