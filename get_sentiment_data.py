import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

path_train = '/Users/yinghua.li/Downloads/archive/twitter_training.csv'
path_validation = '/Users/yinghua.li/Downloads/archive/twitter_validation.csv'


def main():
    df1 = pd.read_csv(path_train, names=['ID', 'entity', 'label', 'text'])
    df2 = pd.read_csv(path_validation, names=['ID', 'entity', 'label', 'text'])
    label_list = list(df1['label'])+list(df2['label'])
    text_list = list(df1['text'])+list(df2['text'])

    df = pd.DataFrame(columns=['text', 'label'])
    df['text'] = text_list
    df['label'] = label_list
    df = df.dropna()

    dic = {'Negative': 0,
           'Positive': 1,
           'Irrelevant': 2,
           'Neutral': 3}

    texts = list(df['text'])
    labels = list(df['label'])
    labels = [dic[i] for i in labels]
    labels = np.array(labels)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=100)

    pickle.dump(x, open('./data/twitter_x.pkl', 'wb'), protocol=4)
    pickle.dump(labels, open('./data/twitter_y.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    main()
