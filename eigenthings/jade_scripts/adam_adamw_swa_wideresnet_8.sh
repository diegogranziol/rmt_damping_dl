#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=6WideResNet28x10

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin


# AdamW

# Weight decay = 1e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/WideResNet28x10/AdamW/wd1e-5_lr_1e-3/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model WideResNet28x10 --lr_init 0.001 --wd 0.01 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/WideResNet28x10/AdamW/wd5e-5_lr_1e-3/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model WideResNet28x10 --lr_init 0.001 --wd 0.05 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/WideResNet28x10/AdamW/wd1e-4_lr_1e-3/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model WideResNet28x10 --lr_init 0.001 --wd 0.1 --epochs 300 --save_freq=5 --decoupled_wd 
