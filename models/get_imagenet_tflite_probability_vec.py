import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


path_model = './onDevice_models/imagenet_inceptionv3.tflite'
data_path = '/raid/yinghua/PriCod/data'
path_tflite_pre_save = '/raid/yinghua/PriCod/data/pkl_data/InceptionV3_pre_tflite.pkl'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    path_list.append(file_absolute_path)
    return path_list


def main():
    path_list = get_path(data_path)
    path_list = sorted(path_list)

    interpreter = tf.lite.Interpreter(model_path=path_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pre_list = []
    i = 0
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        pre = list(output_data)
        pre_list.append(pre)
        print(i)
        i += 1

    pre_np = np.array(pre_list)
    pickle.dump(pre_np, open(path_tflite_pre_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


