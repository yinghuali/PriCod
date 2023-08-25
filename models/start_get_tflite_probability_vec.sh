
python get_tflite_probability_vec.py --path_model './onDevice_models/cifa10_vgg_20.tflite' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/cifa10_alexnet_35.tflite' --path_x '../data/cifar10_x.pkl' --path_save './onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/fmnist_lenet1_3.tflite' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/fmnist_lenet5_3.tflite' --path_x '../data/fashionMnist_x.pkl' --path_save './onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl'

python get_tflite_probability_vec.py --path_model './onDevice_models/plant_nin_6.tflite' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_nin_6_tflite_vec.pkl'
python get_tflite_probability_vec.py --path_model './onDevice_models/plant_vgg19_20.tflite' --path_x '../data/plant_x.pkl' --path_save './onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl'


