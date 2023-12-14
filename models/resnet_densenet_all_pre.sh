

#python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../data/cifar100_x.pkl' --path_save './onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_width_shift_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_width_shift_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_height_shift_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_height_shift_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_horizontal_flip_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_horizontal_flip_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_vertical_flip_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_vertical_flip_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_rotation_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_rotation_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_brightness_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_brightness_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_zoom_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_zoom_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_featurewise_std_normalization_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_featurewise_std_normalization_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_zca_whitening_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_zca_whitening_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_shear_range_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_shear_range_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/augmentation_channel_shift_range_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_augmentation_channel_shift_range_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/noise_salt_pepper_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_noise_salt_pepper_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/noise_gasuss_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_noise_gasuss_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/fog_x.pkl' --path_save './cifar100_ResNet152_1_onDevice_noise_out_vec_tflite/fog_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../noisedata/cifar100/contrast_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_ResNet152_1_contrast_x.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../advdata/cifar100/bim_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_ResNet152_1_bim_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../advdata/cifar100/fsgm_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_ResNet152_1_fsgm_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../advdata/cifar100/patch_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_ResNet152_1_patch_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_ResNet152_1.mlmodel' --model_name 'resnet' --path_x '../advdata/cifar100/pgd_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_ResNet152_1_pgd_x_adv.pkl'

######
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../data/cifar100_x.pkl' --path_save './onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_width_shift_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_width_shift_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_height_shift_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_height_shift_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_horizontal_flip_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_horizontal_flip_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_vertical_flip_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_vertical_flip_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_rotation_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_rotation_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_brightness_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_brightness_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_zoom_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_zoom_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_featurewise_std_normalization_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_featurewise_std_normalization_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_zca_whitening_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_zca_whitening_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_shear_range_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_shear_range_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/augmentation_channel_shift_range_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_augmentation_channel_shift_range_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/noise_salt_pepper_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_noise_salt_pepper_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/noise_gasuss_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_noise_gasuss_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/fog_x.pkl' --path_save './cifar100_DenseNet201_12_onDevice_noise_out_vec_tflite/fog_x.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../noisedata/cifar100/contrast_x.pkl' --path_save './onDevice_noise_out_vec_tflite/cifar100_DenseNet201_12_contrast_x.pkl'

python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../advdata/cifar100/bim_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_DenseNet201_12_bim_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../advdata/cifar100/fsgm_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_DenseNet201_12_fsgm_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../advdata/cifar100/patch_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_DenseNet201_12_patch_x_adv.pkl'
python get_coreml_probability_vec.py --path_model './onDevice_models/cifar100_DenseNet201_12.mlmodel' --model_name 'densenet' --path_x '../advdata/cifar100/pgd_x_adv.pkl' --path_save './onDevice_adv_out_vec_tflite/cifar100_DenseNet201_12_pgd_x_adv.pkl'









