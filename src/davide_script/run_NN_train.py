#!/bin/sh
#SBATCH --gres=gpu:tesla:4         # GPUs requested
#SBATCH --partition=dvd_usr_prod   # partition selected
#SBATCH --account=cin_powerdam_5    # account selected
#SBATCH -N 1      # nodes requested
#SBATCH
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=100000  # memory in Mb
# --open-mode=append
# --open-mode=truncate
#SBATCH -o logs/outfile
#SBATCH -e logs/errfile
#SBATCH -t 05:00:00  # time requested in hour:minute:second

#python train_ae_davide_withImgGen.py
#python train_vae_davide_withImgGen.py
#python train_GANae_davide_withImgGen.py
#python train_GANae_conv_davide_withImgGen.py
python train_DCGan_davide_withImgGen.py
