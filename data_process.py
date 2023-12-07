import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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

    text_list = list(df['text'])
    label_list = list(df['label'])

    dic = {'Negative': 0,
           'Positive': 1,
           'Irrelevant': 2,
           'Neutral': 3}

    y = np.array([dic[i] for i in label_list])
    vectorizer = TfidfVectorizer(max_features=64)
    vectors = vectorizer.fit_transform(text_list)
    x = vectors.toarray()

    pickle.dump(x, open('./data/twitter_x.pkl', 'wb'), protocol=4)
    pickle.dump(y, open('./data/twitter_y.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    main()
