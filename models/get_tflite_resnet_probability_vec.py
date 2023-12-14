import numpy as np
import tensorflow as tf
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_model = args.path_model
path_x = args.path_x
path_save = args.path_save


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
    All_out_probability_vec = np.array(All_out_probability_vec)

    return All_out_probability_vec


def main():
    x = pickle.load(open(path_x, 'rb'))
    x = x.astype('float32')
    if np.max(x) > 5:
        x /= 255.0
    All_out_probability_vec = get_tflite_probability_vec(x)
    pickle.dump(All_out_probability_vec, open(path_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


