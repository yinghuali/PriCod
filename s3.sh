#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python feature_importance.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/importance/plant_nin_6_tflite.json'
python feature_importance.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/importance/plant_nin_6_coreml.json'

python feature_importance.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/importance/plant_vgg19_20_tflite.json'
python feature_importance.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/importance/plant_vgg19_20_coreml.json'


python feature_importance.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/importance/cifar100_ResNet152_1_tflite.json'
python feature_importance.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/importance/cifar100_ResNet152_1_coreml.json'

python feature_importance.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/importance/cifar100_DenseNet201_12_tflite.json'
python feature_importance.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/importance/cifar100_DenseNet201_12_coreml.json'


python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_vgg_20_bim_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_vgg_20_bim_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_vgg_20_bim_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_vgg_20_fsgm_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_vgg_20_fsgm_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_vgg_20_fsgm_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_vgg_20_patch_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_vgg_20_patch_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_vgg_20_patch_x_adv_tflite.json'
python main.py --path_original_out_vec './models/original_adv_out_vec/cifa10_vgg_20_pgd_x_adv.pkl' --path_onDevice_out_vec './models/onDevice_adv_out_vec_tflite/cifa10_vgg_20_pgd_x_adv.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/adv/cifa10_vgg_20_pgd_x_adv_tflite.json'
