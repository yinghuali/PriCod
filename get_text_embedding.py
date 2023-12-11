import numpy as np
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel


path_train = '/home/yinghua/pycharm/PriCod/models/embedding_vec/twitter_training.csv'
path_validation = '/home/yinghua/pycharm/PriCod/models/embedding_vec/twitter_validation.csv'

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        desired_dim = 128
        if embeddings.shape[1] != desired_dim:
            embeddings = torch.nn.functional.pad(embeddings, (0, desired_dim - embeddings.shape[1]))
        embeddings = embeddings.cpu().numpy()
        embeddings = list(embeddings[0])
        return embeddings


def main():
    df1 = pd.read_csv(path_train, names=['ID', 'entity', 'label', 'text'])
    df2 = pd.read_csv(path_validation, names=['ID', 'entity', 'label', 'text'])
    label_list = list(df1['label'])+list(df2['label'])
    text_list = list(df1['text'])+list(df2['text'])

    df = pd.DataFrame(columns=['text', 'label'])
    df['text'] = text_list
    df['label'] = label_list
    df = df.dropna()

    texts_list = list(df['text'])
    embedding_list = []
    n = 0
    for text in texts_list:
        embedding = get_text_embedding(text)
        embedding_list.append(embedding)
        print(n)
        n += 1
    embedding_np = np.array(embedding_list)

    pickle.dump(embedding_np, open('./models/embedding_vec/twitter_embedding.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    main()
