import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_x = '../data/news_x.pkl'
path_y = '../data/news_y.pkl'
num_classes = 20
epochs = 4
batch_size = 128
path_save = './original_models/news_gru_4.h5'


def gru():
    model = Sequential()
    model.add(Embedding(10000, 64, input_length=100))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_one_hot(y, num_classes):
    one_hot_labels = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot_labels[i, label] = 1
    return one_hot_labels


def main():
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    y = get_one_hot(y, num_classes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    model = gru()
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_test, y_test))

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save(path_save)


if __name__ == '__main__':
    main()