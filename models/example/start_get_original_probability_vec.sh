

python get_original_probability_vec.py --path_model './original_models/cifa10_vgg_20.h5' --path_x '../data/cifar10_x.pkl' --path_save './original_out_vec/cifa10_vgg_20_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/cifa10_alexnet_35.h5' --path_x '../data/cifar10_x.pkl' --path_save './original_out_vec/cifa10_alexnet_35_orginal_vec.pkl'

python get_original_probability_vec.py --path_model './original_models/fmnist_lenet1_3.h5' --path_x '../data/fashionMnist_x.pkl' --path_save './original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/fmnist_lenet5_3.h5' --path_x '../data/fashionMnist_x.pkl' --path_save './original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl'

python get_original_probability_vec.py --path_model './original_models/plant_nin_6.h5' --path_x '../data/plant_x.pkl' --path_save './original_out_vec/plant_nin_6_orginal_vec.pkl'
python get_original_probability_vec.py --path_model './original_models/plant_vgg19_20.h5' --path_x '../data/plant_x.pkl' --path_save './original_out_vec/plant_vgg19_20_orginal_vec.pkl'


python get_lstm_original_probability_vec.py --path_model './original_models/twitter_lstm_2.h5' --path_x '../data/twitter_x.pkl' --path_save './original_out_vec/twitter_lstm_2_orginal_vec.pkl'
python get_lstm_original_probability_vec.py --path_model './original_models/twitter_gru_2.h5' --path_x '../data/twitter_x.pkl' --path_save './original_out_vec/twitter_gru_2_orginal_vec.pkl'

