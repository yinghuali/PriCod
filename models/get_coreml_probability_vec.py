import numpy as np
import coremltools
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_model = args.path_model
model_name = args.model_name
path_x = args.path_x
path_save = args.path_save


def get_coreml_probability_vec(x):
    model = coremltools.models.MLModel(path_model)
    All_out_probability_vec = []
    for i in range(len(x)):
        input_data = np.expand_dims(x[i], axis=0)
        if model_name=='vgg':
            predict = list(model.predict({'conv2d_input': input_data})['Identity'][0])
        if model_name=='lenet1':
            predict = list(model.predict({'conv2d_input': input_data})['Identity'][0])
        All_out_probability_vec.append(predict)
    All_out_probability_vec = np.array(All_out_probability_vec)

    return All_out_probability_vec


def main():
    x = pickle.load(open(path_x, 'rb'))
    x = x.astype('float32')
    x /= 255.0
    All_out_probability_vec = get_coreml_probability_vec(x)
    pickle.dump(All_out_probability_vec, open(path_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


