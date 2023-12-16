#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_DenseNet201_12_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_DenseNet201_12_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_DenseNet201_12_bim_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_DenseNet201_12_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_DenseNet201_12_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_DenseNet201_12_fsgm_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_DenseNet201_12_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_DenseNet201_12_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_DenseNet201_12_patch_x_adv_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_adv_out_vec/cifar100_DenseNet201_12_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_coreml/cifar100_DenseNet201_12_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/adv/cifar100_DenseNet201_12_pgd_x_adv_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/combination/cifa10_vgg_20_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/combination/cifa10_vgg_20_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/combination/cifa10_alexnet_35_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/combination/cifa10_alexnet_35_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/combination/fmnist_lenet1_3_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/combination/fmnist_lenet1_3_coreml.json'
