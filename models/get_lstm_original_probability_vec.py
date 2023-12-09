import pickle
import numpy as np
import argparse
import os
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ap = argparse.ArgumentParser()
ap.add_argument("--path_model", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_model = args.path_model
path_x = args.path_x
path_save = args.path_save


def main():
    x = pickle.load(open(path_x, 'rb'))
    x = x.astype('float32')
    original_model = load_model(path_model)
    ori_probabilities = original_model.predict(x)
    pickle.dump(ori_probabilities, open(path_save, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


