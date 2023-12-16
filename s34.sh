#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p batch
#SBATCH --mem 10G

python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_width_shift_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_width_shift_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_width_shift_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_height_shift_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_height_shift_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_height_shift_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_horizontal_flip_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_horizontal_flip_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_horizontal_flip_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_vertical_flip_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_vertical_flip_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_vertical_flip_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_rotation_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_rotation_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_rotation_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_brightness_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_brightness_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_brightness_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_zoom_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_zoom_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_zoom_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_featurewise_std_normalization_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_featurewise_std_normalization_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_featurewise_std_normalization_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_zca_whitening_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_zca_whitening_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_zca_whitening_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_shear_range_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_shear_range_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_shear_range_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_augmentation_channel_shift_range_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_channel_shift_range_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_augmentation_channel_shift_range_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_noise_salt_pepper_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_noise_salt_pepper_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_noise_salt_pepper_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_noise_gasuss_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_noise_gasuss_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_noise_gasuss_tflite.json'
python main_pfd.py --path_original_out_vec './models/original_noise_out_vec/cifar100_ResNet152_1_contrast_x.pkl' --path_onDevice_out_vec './models/onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_contrast_x.pkl' --path_embedding_vec './models/embedding_vec/cifar100_embedding.pkl' --path_y './data/cifar100_y.pkl' --path_save_res './results/noise/cifar100_ResNet152_1_contrast_tflite.json'
