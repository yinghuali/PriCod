import numpy as np
import pandas as pd
import pickle


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

    texts = list(df['text'])
    print(len(texts))


    # pickle.dump(x, open('./models/embedding_vec/twitter_embedding.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    main()
