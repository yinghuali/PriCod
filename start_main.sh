#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_vgg_20_tflite.json'
python main.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_vgg_20_coreml.json'

python main.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_alexnet_35_tflite.json'
python main.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_alexnet_35_coreml.json'

python main.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet1_3_tflite.json'
python main.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet1_3_coreml.json'


