

python get_coreml_probability_vec.py --path_model './onDevice_models/cifa10_vgg_20.mlmodel' --model_name 'vgg' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifa10_alexnet_35.mlmodel' --model_name 'alexnet' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/fmnist_lenet1_3.mlmodel' --model_name 'lenet1' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/fmnist_lenet5_3.mlmodel' --model_name 'lenet5' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet5_3_coreml_vec.pkl'


python get_coreml_probability_vec.py --path_model './onDevice_models/plant_nin_6.mlmodel' --model_name 'nin' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_nin_6_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/plant_vgg19_20.mlmodel' --model_name 'vgg19' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_vgg19_20_coreml_vec.pkl'


python get_lstm_coreml_probability_vec.py --path_model './onDevice_models/news_lstm_4.mlmodel' --model_name 'lstm' --path_x '../data/news_x.pkl' --path_save './onDevice_out_vec/news_lstm_4_coreml_vec.pkl'
python get_lstm_coreml_probability_vec.py --path_model './onDevice_models/news_gru_4.mlmodel' --model_name 'lstm' --path_x '../data/news_x.pkl' --path_save './onDevice_out_vec/news_gru_4_coreml_vec.pkl'


python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../data/cifar100_x.pkl' --path_save './onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../data/cifar100_x.pkl' --path_save './onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl'

