#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python get_tflite_probability_vec.py --path_tflite './onDevice_models/cifa10_vgg_20.tflite' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl'

