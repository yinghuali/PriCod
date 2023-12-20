import pickle
from diff_feature import *
from sklearn.model_selection import train_test_split


path_y_list = ['data/cifar10_y.pkl', 'data/cifar10_y.pkl', 'data/cifar10_y.pkl', 'data/cifar10_y.pkl',
               'data/fashionMnist_y.pkl', 'data/fashionMnist_y.pkl', 'data/fashionMnist_y.pkl', 'data/fashionMnist_y.pkl',
               'data/plant_y.pkl', 'data/plant_y.pkl', 'data/plant_y.pkl', 'data/plant_y.pkl',
               'data/cifar100_y.pkl', 'data/cifar100_y.pkl',
               'data/news_y.pkl', 'data/news_y.pkl',
               ]
path_original_out_vec_list = ['models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl', 'models/original_out_vec/cifa10_vgg_20_orginal_vec.pkl',
                              'models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl', 'models/original_out_vec/cifa10_alexnet_35_orginal_vec.pkl',
                              'models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl', 'models/original_out_vec/fashionMnist_lenet1_3_orginal_vec.pkl',
                              'models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl', 'models/original_out_vec/fashionMnist_lenet5_3_orginal_vec.pkl',
                              'models/original_out_vec/plant_nin_6_orginal_vec.pkl', 'models/original_out_vec/plant_nin_6_orginal_vec.pkl',
                              'models/original_out_vec/plant_vgg19_20_orginal_vec.pkl', 'models/original_out_vec/plant_vgg19_20_orginal_vec.pkl',
                              'models/original_out_vec/cifar100_DenseNet201_12_orginal_vec.pkl', 'models/original_out_vec/cifar100_ResNet152_1_orginal_vec.pkl',
                              'models/original_out_vec/news_gru_4_orginal_vec.pkl', 'models/original_out_vec/news_lstm_4_orginal_vec.pkl'


                         ]


path_onDevice_out_vec_list = ['models/onDevice_out_vec/cifa10_vgg_20_tflite_vec.pkl', 'models/onDevice_out_vec/cifa10_vgg_20_coreml_vec.pkl',
                              'models/onDevice_out_vec/cifa10_alexnet_35_tflite_vec.pkl', 'models/onDevice_out_vec/cifa10_alexnet_35_coreml_vec.pkl',
                              'models/onDevice_out_vec/fashionMnist_lenet1_3_tflite_vec.pkl', 'models/onDevice_out_vec/fashionMnist_lenet1_3_coreml_vec.pkl',
                              'models/onDevice_out_vec/fashionMnist_lenet5_3_tflite_vec.pkl', 'models/onDevice_out_vec/fashionMnist_lenet5_3_coreml_vec.pkl',
                              'models/onDevice_out_vec/plant_nin_6_tflite_vec.pkl', 'models/onDevice_out_vec/plant_nin_6_coreml_vec.pkl',
                              'models/onDevice_out_vec/plant_vgg19_20_tflite_vec.pkl', 'models/onDevice_out_vec/plant_vgg19_20_coreml_vec.pkl',

                              'models/onDevice_out_vec/cifar100_DenseNet201_12_coreml_vec.pkl', 'models/onDevice_out_vec/cifar100_ResNet152_1_coreml_vec.pkl',
                              'models/onDevice_out_vec/news_gru_4_coreml_vec.pkl', 'models/onDevice_out_vec/news_lstm_4_coreml_vec.pkl'
                         ]


def get_top_miss(distance, test_y, onDevice_pre_y):
    sorted_indices = np.argsort(distance)[::-1]
    miss_idx = []
    for i in range(len(test_y)):
        if onDevice_pre_y[i] != test_y[i]:
            miss_idx.append(i)
    miss_idx = np.array(miss_idx)
    t = int(len(test_y)/10)

    top_0t_to_1t_indices = sorted_indices[:t]
    top_1t_to_2t_indices = sorted_indices[t:t*2]
    top_2t_to_3t_indices = sorted_indices[t*2:t*3]
    top_3t_to_4t_indices = sorted_indices[t*3:t*4]
    top_4t_to_5t_indices = sorted_indices[t*4:t*5]
    top_5t_to_6t_indices = sorted_indices[t*5:t*6]
    top_6t_to_7t_indices = sorted_indices[t*6:t*7]
    top_7t_to_8t_indices = sorted_indices[t*7:t*8]
    top_8t_to_9t_indices = sorted_indices[t*8:t*9]
    top_9t_to_10t_indices = sorted_indices[t*9:t*10]

    dic = {}
    dic['top_0t_to_1t'] = len(np.intersect1d(top_0t_to_1t_indices, miss_idx))
    dic['top_1t_to_2t'] = len(np.intersect1d(top_1t_to_2t_indices, miss_idx))
    dic['top_2t_to_3t'] = len(np.intersect1d(top_2t_to_3t_indices, miss_idx))
    dic['top_3t_to_4t'] = len(np.intersect1d(top_3t_to_4t_indices, miss_idx))
    dic['top_4t_to_5t'] = len(np.intersect1d(top_4t_to_5t_indices, miss_idx))
    dic['top_5t_to_6t'] = len(np.intersect1d(top_5t_to_6t_indices, miss_idx))
    dic['top_6t_to_7t'] = len(np.intersect1d(top_6t_to_7t_indices, miss_idx))
    dic['top_7t_to_8t'] = len(np.intersect1d(top_7t_to_8t_indices, miss_idx))
    dic['top_8t_to_9t'] = len(np.intersect1d(top_8t_to_9t_indices, miss_idx))
    dic['top_9t_to_10t'] = len(np.intersect1d(top_9t_to_10t_indices, miss_idx))

    return dic


def main(path_y_list, path_original_out_vec_list, path_onDevice_out_vec_list):

    value_list = []
    for i in range(len(path_y_list)):
        path_y = path_y_list[i]
        path_original_out_vec = path_original_out_vec_list[i]
        path_onDevice_out_vec = path_onDevice_out_vec_list[i]
        original_out_vec = pickle.load(open(path_original_out_vec, 'rb'))
        onDevice_out_vec = pickle.load(open(path_onDevice_out_vec, 'rb'))

        y = pickle.load(open(path_y, 'rb'))
        if y.shape == (y.size,):
            y = y
        else:
            y = np.array([i[0] for i in y])

        original_out_vec_train, original_out_vec_test, train_y, test_y = train_test_split(original_out_vec, y, test_size=0.3, random_state=0)
        onDevice_out_vec_train, onDevice_out_vec_test, _, _ = train_test_split(onDevice_out_vec, y, test_size=0.3, random_state=0)

        onDevice_pre_y = np.argmax(onDevice_out_vec_test, axis=1)

        # distance = euclidean_distance(original_out_vec_test, onDevice_out_vec_test)

        # distance = manhattan_distance(original_out_vec_test, onDevice_out_vec_test)
        # distance = chebyshev_distance(original_out_vec_test, onDevice_out_vec_test)
        # distance = sum_squared_differences(original_out_vec_test, onDevice_out_vec_test)
        distance = wasserstein(original_out_vec_test, onDevice_out_vec_test)


        dic = get_top_miss(distance, test_y, onDevice_pre_y)
        key_list = ['top_0t_to_1t', 'top_1t_to_2t', 'top_2t_to_3t', 'top_3t_to_4t', 'top_4t_to_5t', 'top_5t_to_6t', 'top_6t_to_7t', 'top_7t_to_8t', 'top_8t_to_9t', 'top_9t_to_10t']
        value = [dic[i] for i in key_list]
        value_list.append(value)
    value_np = np.array(value_list)
    value_final = np.mean(value_np, axis=0)
    value_final = [int(i) for i in value_final]
    print(value_final)



if __name__ == '__main__':
    main(path_y_list, path_original_out_vec_list, path_onDevice_out_vec_list)


