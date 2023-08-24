

python vgg.py --path_x '../data/cifar10_x.pkl' --path_y '../data/cifar10_y.pkl' --num_classes 10 --epochs 20 --batch_size 64 --path_save './original_models/cifa10_vgg_20.h5'
python alexnet.py --path_x '../data/cifar10_x.pkl' --path_y '../data/cifar10_y.pkl' --num_classes 10 --epochs 35 --batch_size 64 --path_save './original_models/cifa10_alexnet_35.h5'

python lenet1.py --path_x '../data/fashionMnist_x.pkl' --path_y '../data/fashionMnist_y.pkl' --num_classes 10 --epochs 3 --batch_size 64 --path_save './original_models/fmnist_lenet1_3.h5'
python lenet5.py --path_x '../data/fashionMnist_x.pkl' --path_y '../data/fashionMnist_y.pkl' --num_classes 10 --epochs 3 --batch_size 64 --path_save './original_models/fmnist_lenet5_3.h5'


python resnet.py --path_x '../data/imagenet_x.pkl' --path_y '../data/imagenet_y.pkl' --num_classes 200 --epochs 70 --batch_size 64 --path_save './original_models/imagenet_resnet_70.h5'

