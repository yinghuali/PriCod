import pickle
import json
import argparse
from diff_feature import get_all_feature
from get_rank_idx import *
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

ap = argparse.ArgumentParser()

ap.add_argument("--path_original_out_vec", type=str)
ap.add_argument("--path_onDevice_out_vec", type=str)
ap.add_argument("--path_embedding_vec", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_save_res", type=str)
args = ap.parse_args()


path_original_out_vec = args.path_original_out_vec
path_onDevice_out_vec = args.path_onDevice_out_vec
path_embedding_vec = args.path_embedding_vec
path_y = args.path_y
path_save_res = args.path_save_res

original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
embedding_vec = pickle.load(open(path_embedding_vec, 'rb'))


def get_add(x1, x2):
    x = x1 + x2
    return x


def get_multiplication(x1, x2):
    x = x1 * x2
    return x


def get_cross_fusion(x1, x2):
    f1 = x1
    f2 = x2
    f3 = x1*x1
    f4 = x2*x2
    f5 = x1*x2
    x = np.hstack((f1, f2, f3, f4, f5))
    return x


def get_embedding_PowerTransformer(x1, x2):
    x = np.hstack((x1, x2))
    pt = preprocessing.PowerTransformer(method="yeo-johnson")
    x = pt.fit_transform(x)
    return x


def get_QuantileTransformer(x1, x2):
    x = np.hstack((x1, x2))
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    x = quantile_transformer.fit_transform(x)
    return x


def main():
    y = pickle.load(open(path_y, 'rb'))
    y = np.array([i[0] for i in y])
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

    distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)
    n_diff = len(distance_feature[0])

    pca = PCA(n_components=n_diff)
    new_embedding_vec = pca.fit_transform(embedding_vec)

    PowerTransformer_feature = get_embedding_PowerTransformer(distance_feature, embedding_vec)
    QuantileTransformer_feature = get_QuantileTransformer(distance_feature, embedding_vec)
    CrossFusion_feature = get_cross_fusion(distance_feature, new_embedding_vec)
    Multiplication_feature = get_multiplication(distance_feature, new_embedding_vec)
    Add_feaure = get_add(distance_feature, new_embedding_vec)

    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)
    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    Add_feature_train, Add_feature_test, _, _ = train_test_split(Add_feaure, y, test_size=0.3, random_state=0)
    Multiplication_feature_train, Multiplication_feature_test, _, _ = train_test_split(Multiplication_feature, y, test_size=0.3, random_state=0)
    CrossFusion_feature_train, CrossFusion_feature_test, _, _ = train_test_split(CrossFusion_feature, y, test_size=0.3, random_state=0)
    QuantileTransformer_feature_train, QuantileTransformer_feature_test, _, _ = train_test_split(QuantileTransformer_feature, y, test_size=0.3, random_state=0)
    PowerTransformer_feature_train, PowerTransformer_feature_test, _, _ = train_test_split(PowerTransformer_feature, y, test_size=0.3, random_state=0)

    dic = {}
    model = LGBMClassifier()
    model.fit(Add_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(Add_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    Add_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['Add'] = Add_apfd

    model = LGBMClassifier()
    model.fit(Multiplication_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(Multiplication_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    Multiplication_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['Multiplication'] = Multiplication_apfd

    model = LGBMClassifier()
    model.fit(CrossFusion_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(CrossFusion_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    CrossFusion_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['CrossFusion'] = CrossFusion_apfd

    model = LGBMClassifier()
    model.fit(QuantileTransformer_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(QuantileTransformer_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    QuantileTransformer_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['QuantileTransformer'] = QuantileTransformer_apfd

    model = LGBMClassifier()
    model.fit(PowerTransformer_feature_train, miss_train_label)
    y_concat_all = model.predict_proba(PowerTransformer_feature_test)[:, 1]
    rank_idx = y_concat_all.argsort()[::-1].copy()
    PowerTransformer_apfd = apfd(idx_miss_test_list, rank_idx)
    dic['PowerTransformer'] = PowerTransformer_apfd

    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)


if __name__ == '__main__':
    main()
