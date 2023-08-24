import os
import numpy as np
import cv2
import pickle
from tensorflow.keras.datasets import cifar10, fashion_mnist


def get_path_list(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cifar10_x = np.vstack((x_train, x_test))
    cifar10_y = np.vstack((y_train, y_test))
    pickle.dump(cifar10_x, open('./data/cifar10_x.pkl', 'wb'), protocol=4)
    pickle.dump(cifar10_y, open('./data/cifar10_y.pkl', 'wb'), protocol=4)


def get_imagenet(path_data_dir):
    path_list = get_path_list(path_data_dir)
    label_list = [i.split('/train/')[-1].split('/')[0].strip() for i in path_list]
    label_set = list(set(label_list))
    dic = dict(zip(label_set, range(len(label_set))))
    label_np = np.array([[dic[i]] for i in label_list])
    img_np = []
    for i in path_list:
        img = cv2.imread(i)
        img_np.append(img)
    img_np = np.array(img_np)
    img_np = img_np.astype(np.int8)
    print(img_np.shape)
    print(label_np.shape)

    pickle.dump(img_np, open('./data/imagenet_x.pkl', 'wb'), protocol=4)
    pickle.dump(label_np, open('./data/imagenet_y.pkl', 'wb'), protocol=4)


def get_Fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_train = np.array([[i] for i in y_train])
    y_test = np.array([[i] for i in y_test])
    fashionMnist_x = np.vstack((x_train, x_test))
    fashionMnist_y = np.vstack((y_train, y_test))
    pickle.dump(fashionMnist_x, open('./data/fashionMnist_x.pkl', 'wb'), protocol=4)
    pickle.dump(fashionMnist_y, open('./data/fashionMnist_y.pkl', 'wb'), protocol=4)


def main():
    get_cifar10()
    get_imagenet('./data/imagenet/tiny-imagenet-200/train/')
    get_Fashion()


if __name__ == '__main__':
    main()


