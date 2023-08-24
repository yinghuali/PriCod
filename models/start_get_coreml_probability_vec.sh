

python get_coreml_probability_vec.py --path_model './onDevice_models/cifa10_vgg_20.mlmodel' --model_name 'vgg' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifa10_alexnet_35.mlmodel' --model_name 'vgg' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/fmnist_lenet1_3.mlmodel' --model_name 'lenet1' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/fmnist_lenet5_3.mlmodel' --model_name 'lenet5' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet5_3_coreml_vec.pkl'
