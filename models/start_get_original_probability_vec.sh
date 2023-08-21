#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch
#SBATCH --mem 10G

python get_original_probability_vec.py --path_model './orginal_models/cifa10_vgg_20.h5' --path_x '../data/cifar10_x.pkl' --path_save './original_out_vec/cifa10_vgg_20_orginal_vec.pkl'

