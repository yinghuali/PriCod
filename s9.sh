#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet1_3_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet1_3_bim_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet1_3_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet1_3_fsgm_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet1_3_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet1_3_patch_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet1_3_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet1_3_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet1_3_pgd_x_adv_coreml.json'


python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet5_3_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet5_3_bim_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet5_3_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet5_3_fsgm_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet5_3_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet5_3_patch_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/fmnist_lenet5_3_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/fmnist_lenet5_3_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/adv/fmnist_lenet5_3_pgd_x_adv_coreml.json'

python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_ResNet152_1_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_ResNet152_1_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_ResNet152_1_bim_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_ResNet152_1_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_ResNet152_1_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_ResNet152_1_fsgm_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_ResNet152_1_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_ResNet152_1_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_ResNet152_1_patch_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_ResNet152_1_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_ResNet152_1_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_ResNet152_1_pgd_x_adv_coreml.json'
