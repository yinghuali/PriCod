python convert2tflite.py --path_model './original_models/cifa10_vgg_20.h5' --path_save './onDevice_models/cifa10_vgg_20.tflite'
python convert2coreml.py --path_model './original_models/cifa10_vgg_20.h5' --path_save './onDevice_models/cifa10_vgg_20.mlmodel'


python convert2tflite.py --path_model './original_models/cifa10_alexnet_35.h5' --path_save './onDevice_models/cifa10_alexnet_35.tflite'
python convert2coreml.py --path_model './original_models/cifa10_alexnet_35.h5' --path_save './onDevice_models/cifa10_alexnet_35.mlmodel'


python convert2tflite.py --path_model './original_models/fmnist_lenet1_3.h5' --path_save './onDevice_models/fmnist_lenet1_3.tflite'
python convert2coreml.py --path_model './original_models/fmnist_lenet1_3.h5' --path_save './onDevice_models/fmnist_lenet1_3.mlmodel'

python convert2tflite.py --path_model './original_models/fmnist_lenet5_3.h5' --path_save './onDevice_models/fmnist_lenet5_3.tflite'
python convert2coreml.py --path_model './original_models/fmnist_lenet5_3.h5' --path_save './onDevice_models/fmnist_lenet5_3.mlmodel'

python convert2tflite.py --path_model './original_models/plant_resnet_5.h5' --path_save './onDevice_models/plant_resnet_5.tflite'
python convert2coreml.py --path_model './original_models/plant_resnet_5.h5' --path_save './onDevice_models/plant_resnet_5.mlmodel'

