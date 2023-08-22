python convert2tflite.py --path_model './original_models/cifa10_vgg_20.h5' --path_save'./onDevice_models/cifa10_vgg_20.tflite'

python convert2coreml.py --path_model './original_models/cifa10_vgg_20.h5' --path_save './onDevice_models/cifa10_vgg_20.mlmodel'