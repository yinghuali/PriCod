from ATS import ATS
import pickle
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()

ap.add_argument("--nb_classes", type=int)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_onDevice_out_vec", type=str)
ap.add_argument("--subject_name", type=str)
args = ap.parse_args()

nb_classes = args.nb_classes
path_x = args.path_x
path_y = args.path_y
path_onDevice_out_vec = args.path_onDevice_out_vec
save_subject_name = args.subject_name

# nb_classes = 10
# path_y = '../data/cifar10_y.pkl'
# path_onDevice_out_vec = '../models/onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl'
# save_subject_name = 'original_cifa10_alexnet_35_coreml'


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def get_miss_lable(target_train_pre, target_test_pre, y_train, y_test):
    idx_miss_train_list = get_idx_miss_class(target_train_pre, y_train)
    idx_miss_test_list = get_idx_miss_class(target_test_pre, y_test)
    miss_train_label = [0]*len(y_train)
    for i in idx_miss_train_list:
        miss_train_label[i]=1
    miss_train_label = np.array(miss_train_label)

    miss_test_label = [0]*len(y_test)
    for i in idx_miss_test_list:
        miss_test_label[i]=1
    miss_test_label = np.array(miss_test_label)

    return miss_train_label, miss_test_label, idx_miss_test_list


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd


def main():
    y = pickle.load(open(path_y, 'rb'))
    if y.shape == (y.size,):
        y = y
    else:
        y = np.array([i[0] for i in y])

    onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))
    onDevice_pre_y = onDevice_out_vec.argmax(axis=1)

    target_train_pre, target_test_pre, train_y, test_y = train_test_split(onDevice_pre_y, y, test_size=0.3, random_state=0)
    onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)
    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    ats = ATS()
    div_rank, _, _ = ats.get_priority_sequence(onDevice_out_vec_test, target_test_pre, nb_classes, onDevice_out_vec_test, th=0.001)
    ast_rank_idx = div_rank
    ast_apfd = apfd(idx_miss_test_list, ast_rank_idx)
    print(ast_apfd)
    write_result(save_subject_name+'->'+str(ast_apfd), 'result.txt')


if __name__ == '__main__':
    main()
