#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_width_shift_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_width_shift_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_width_shift_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_height_shift_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_height_shift_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_height_shift_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_horizontal_flip_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_horizontal_flip_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_horizontal_flip_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_vertical_flip_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_vertical_flip_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_vertical_flip_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_rotation_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_rotation_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_rotation_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_brightness_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_brightness_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_brightness_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_zoom_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_zoom_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_zoom_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_featurewise_std_normalization_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_featurewise_std_normalization_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_featurewise_std_normalization_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_zca_whitening_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_zca_whitening_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_zca_whitening_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_shear_range_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_shear_range_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_shear_range_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_augmentation_channel_shift_range_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_augmentation_channel_shift_range_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_augmentation_channel_shift_range_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_noise_salt_pepper_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_noise_salt_pepper_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_noise_salt_pepper_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_noise_gasuss_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_noise_gasuss_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_noise_gasuss_coreml.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/fmnist_lenet5_3_contrast_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_coreml/fmnist_lenet5_3_contrast_x.pkl' --path_embedding_vec './models/embedding_vec/fmnist_embedding.pkl' --path_y './data/fashionMnist_y.pkl' --path_save_res './results_pfd/noise/fmnist_lenet5_3_contrast_coreml.json'
