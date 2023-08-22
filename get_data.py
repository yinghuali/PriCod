import numpy as np
import pickle
from tensorflow.keras.datasets import cifar10, cifar100


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cifar10_x = np.vstack((x_train, x_test))
    cifar10_y = np.vstack((y_train, y_test))
    pickle.dump(cifar10_x, open('./data/cifar10_x.pkl', 'wb'), protocol=4)
    pickle.dump(cifar10_y, open('./data/cifar10_y.pkl', 'wb'), protocol=4)


def get_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cifar100_x = np.vstack((x_train, x_test))
    cifar100_y = np.vstack((y_train, y_test))
    pickle.dump(cifar100_x, open('./data/cifar100_x.pkl', 'wb'), protocol=4)
    pickle.dump(cifar100_y, open('./data/cifar100_y.pkl', 'wb'), protocol=4)


def main():
    # get_cifar10()
    # get_cifar100()
    pass


if __name__ == '__main__':
    main()


