import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path_original_out_vec = './models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl'
path_onDevice_out_vec = './models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl'
path_y = './data/cifar10_y.pkl'

original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
y = pickle.load(open(path_y, 'rb'))
y = np.array([i[0] for i in y])

original_pre_y = original_out_vec.argmax(axis=1)
onDevice_pre_y = onDevice_out_vec.argmax(axis=1)


def get_kill_feature(original_out_vec, onDevice_out_vec):
    kill_feaure = []
    original_pre_y = original_out_vec.argmax(axis=1)
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)
    for i in range(len(original_pre_y)):
        if original_pre_y[i] != onDevice_pre_y[i]:
            kill_feaure.append(1)
        else:
            kill_feaure.append(0)
    kill_feaure = np.array(kill_feaure)
    return kill_feaure


# def get_confidence_feature(original_out_vec, onDevice_out_vec):
#     confidence_feaure = []
#     original_pre_y = original_out_vec.argmax(axis=1)
#     onDevice_pre_y = onDevice_out_vec.argmax(axis=1)
#     for i in range


