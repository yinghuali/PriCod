#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_alexnet_35_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_alexnet_35_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_alexnet_35_bim_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_alexnet_35_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_alexnet_35_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_alexnet_35_fsgm_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_alexnet_35_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_alexnet_35_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_alexnet_35_patch_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_alexnet_35_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_alexnet_35_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_alexnet_35_pgd_x_adv_tflite.json'


python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet1_3_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet1_3_bim_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet1_3_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet1_3_fsgm_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet1_3_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet1_3_patch_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet1_3_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet1_3_pgd_x_adv_tflite.json'


python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet5_3_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet5_3_bim_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet5_3_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet5_3_fsgm_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet5_3_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet5_3_patch_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/fmnist_lenet5_3_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/adv/fmnist_lenet5_3_pgd_x_adv_tflite.json'
