import pickle
import json
import argparse
from diff_feature import get_all_feature
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from lightgbm import LGBMClassifier
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


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def main():
    y = pickle.load(open(path_y, 'rb'))
    y = np.array([i[0] for i in y])
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

    distance_feature = get_all_feature(original_out_vec, onDevice_out_vec)
    concat_all_feature = np.hstack((distance_feature, onDevice_out_vec, embedding_vec))

    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    concat_train_all_feature, concat_test_all_feature, _, _ = train_test_split(concat_all_feature, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = LGBMClassifier()
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    model_rank_idx = y_concat_all.argsort()[::-1].copy()

    deepGini_rank_idx = DeepGini_rank_idx(onDevice_out_vec_test)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(onDevice_out_vec_test)
    pcs_rank_idx = PCS_rank_idx(onDevice_out_vec_test)
    entropy_rank_idx = Entropy_rank_idx(onDevice_out_vec_test)
    random_rank_idx = Random_rank_idx(onDevice_out_vec_test)

    select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    idx_miss_list = get_idx_miss_class(target_test_pre, test_y)
    deepGini_pfd = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    random_pfd = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)
    vanillasoftmax_pfd = get_res_ratio_list(idx_miss_list, vanillasoftmax_rank_idx, select_ratio_list)
    pcs_pfd = get_res_ratio_list(idx_miss_list, pcs_rank_idx, select_ratio_list)
    entropy_pfd = get_res_ratio_list(idx_miss_list, entropy_rank_idx, select_ratio_list)
    model_pfd = get_res_ratio_list(idx_miss_list, model_rank_idx, select_ratio_list)

    dic = {
        'random_pfd': random_pfd,
        'deepGini_pfd': deepGini_pfd,
        'vanillasoftmax_pfd': vanillasoftmax_pfd,
        'pcs_pfd': pcs_pfd,
        'entropy_pfd': entropy_pfd,
        'model_pfd': model_pfd,
    }

    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)

    print('random_pfd', random_pfd)
    print('deepGini_pfd', deepGini_pfd)
    print('vanillasoftmax_pfd', vanillasoftmax_pfd)
    print('pcs_pfd', pcs_pfd)
    print('entropy_pfd', entropy_pfd)
    print('model_pfd', model_pfd)


if __name__ == '__main__':
    main()
