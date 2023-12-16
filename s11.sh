#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_combination.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/combination/fmnist_lenet5_3_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet5_3_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/combination/fmnist_lenet5_3_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/combination/plant_nin_6_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/combination/plant_nin_6_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/combination/plant_vgg19_20_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/combination/plant_vgg19_20_coreml.json'


python main_combination.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/combination/cifar100_DenseNet201_12_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/combination/cifar100_DenseNet201_12_coreml.json'

python main_combination.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/combination/cifar100_ResNet152_1_tflite.json'
python main_combination.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/combination/cifar100_ResNet152_1_coreml.json'
