#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_vgg_20_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_vgg_20_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_alexnet_35_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar10_embedding.pkl' --path_y './data/cifar10_y.pkl' --path_save_res './results/original/cifa10_alexnet_35_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet1_3_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet1_3_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet5_3_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/fashionMnist_lenet5_3_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results/original/fmnist_lenet5_3_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/original/plant_nin_6_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/plant_nin_6_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_nin_6_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/original/plant_nin_6_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/original/plant_vgg19_20_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/plant_vgg19_20_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/plant_vgg19_20_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/plant_embedding.pkl' --path_y './data/plant_y.pkl' --path_save_res './results/original/plant_vgg19_20_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/original/cifar100_ResNet152_1_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/original/cifar100_ResNet152_1_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/original/cifar100_DenseNet201_12_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/original/cifar100_DenseNet201_12_coreml.json'


python main_uncertainty.py --path_original_out_vec './models/original_out_vec/news_gru_4_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/news_gru_4_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/news_embedding.pkl' --path_y './data/news_y.pkl' --path_save_res './results/combination/news_gru_4_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/news_gru_4_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/news_gru_4_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/news_embedding.pkl' --path_y './data/news_y.pkl' --path_save_res './results/combination/news_gru_4_coreml.json'

python main_uncertainty.py --path_original_out_vec './models/original_out_vec/news_lstm_4_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/news_lstm_4_tflite_vec.pkl' --path_embedding_vec './models/embedding_vec/news_embedding.pkl' --path_y './data/news_y.pkl' --path_save_res './results/combination/news_lstm_4_tflite.json'
python main_uncertainty.py --path_original_out_vec './models/original_out_vec/news_lstm_4_orginal_vec.pkl' --path_onDevice_out_vec './models/onDevice_out_vec/news_lstm_4_coreml_vec.pkl' --path_embedding_vec './models/embedding_vec/news_embedding.pkl' --path_y './data/news_y.pkl' --path_save_res './results/combination/news_lstm_4_coreml.json'