#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH --output=/dev/null
#SBATCH -p batch
#SBATCH --mem 10G

python get_tflite_probability_vec.py --path_model './onDevice_models/cifa10_vgg_20.tflite' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/cifa10_alexnet_35.tflite' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/fmnist_lenet1_3.tflite' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/fmnist_lenet5_3.tflite' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl'

python get_tflite_probability_vec.py --path_model './onDevice_models/plant_nin_6.tflite' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_nin_6_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/plant_vgg19_20.tflite' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl'

python get_lstm_probability_vec.py --path_model './onDevice_models/twitter_lstm_2.tflite' --path_x '../data/twitter_x.pkl' --path_save './onDevice_out_vec/twitter_lstm_2_tflite_vec.pkl'
python get_lstm_probability_vec.py --path_model './onDevice_models/twitter_gru_2.tflite' --path_x '../data/twitter_x.pkl' --path_save './onDevice_out_vec/twitter_gru_2_tflite_vec.pkl'

python get_lstm_probability_vec.py --path_model './onDevice_models/news_lstm_4.tflite' --path_x '../data/news_x.pkl' --path_save './onDevice_out_vec/news_lstm_4_tflite_vec.pkl'
python get_lstm_probability_vec.py --path_model './onDevice_models/news_gru_4.tflite' --path_x '../data/news_x.pkl' --path_save './onDevice_out_vec/news_gru_4_tflite_vec.pkl'

python get_tflite_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.tflite' --path_x '../data/cifar100_x.pkl' --path_save './onDevice_out_vec/cifar100_ResNet152_1_tflite_vec.pkl'


