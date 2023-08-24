#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p gpu
#SBATCH --mem 10G

python get_original_probability_vec.py --path_model './original_models/cifa10_vgg_20.h5' --path_x '../data/cifar10_x.pkl' --path_save './original_out_vec/cifa10_vgg_20_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/cifa10_alexnet_35.h5' --path_x '../data/cifar10_x.pkl' --path_save './original_out_vec/cifa10_alexnet_35_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/fmnist_lenet1_3.h5' --path_x '../data/fashionMnist_x.pkl' --path_save './original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/fmnist_lenet5_3.h5' --path_x '../data/fashionMnist_x.pkl' --path_save './original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl'


