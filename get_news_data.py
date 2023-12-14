import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

path_train = '/Users/yinghua.li/Downloads/archive/train_data.csv'
path_validation = '/Users/yinghua.li/Downloads/archive/valid_data.csv'


def main():
    df1 = pd.read_csv(path_train)
    df2 = pd.read_csv(path_validation)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna()

    labels = np.array(list(df['label']))
    texts = list(df['text'])

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=100)
    print(x.shape)

    pickle.dump(x, open('./data/news_x.pkl', 'wb'), protocol=4)
    pickle.dump(labels, open('./data/news_y.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    main()
