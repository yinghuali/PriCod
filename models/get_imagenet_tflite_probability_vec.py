import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import decode_predictions
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



path_model = './onDevice_models/imagenet_inceptionv3.tflite'
img_path = '/raid/yinghua/PriCod/data/imagenet/train/n04554684/n04554684_6202.JPEG'



def get_tflite_probability_vec(x):
    interpreter = tf.lite.Interpreter(model_path=path_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    All_out_probability_vec = []
    for i in range(len(x)):
        input_data = np.expand_dims(x[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        All_out_probability_vec.append(output_data)
        print('==========', i)

        preds_vec = np.array([output_data])
        pre = decode_predictions(preds_vec, top=1)
        y = pre[0][0][0]
        probability = pre[0][0][2]
        print(y)
        print(probability)

    All_out_probability_vec = np.array(All_out_probability_vec)

    return All_out_probability_vec


def main():
    img = image.load_img(img_path, target_size=(299, 299))
    img = np.array(img)
    img = img.astype('float32')
    img /= 255.0
    x = [img]
    All_out_probability_vec = get_tflite_probability_vec(x)



if __name__ == '__main__':
    main()


