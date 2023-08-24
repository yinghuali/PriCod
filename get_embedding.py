import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path_x", type=str)
ap.add_argument("--save_embedding", type=str)
ap.add_argument("--cuda", type=str)
args = ap.parse_args()

path_x = args.path_x
save_embedding = args.save_embedding
cuda = args.cuda
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

# python get_embedding.py --path_x './data/cifar10_x.pkl' --save_embedding './models/embedding_vec/cifar10_embedding.pkl' --cuda 'cuda:0'
# python get_embedding.py --path_x './data/fashionMnist_x.pkl' --save_embedding './models/embedding_vec/fmnist_embedding.pkl' --cuda 'cuda:0'


def main():
    x = pickle.load(open(path_x, 'rb'))
    x = x.astype('float32')
    x /= 255.0
    if x.shape[-1]==1:
        x = np.repeat(x, repeats=3, axis=3)

    x = np.transpose(x, (0, 3, 1, 2))
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.float()
    model.to(device)
    model.eval()

    x_embedding = []
    for i in range(len(x)):
        input = np.array([x[i]])
        input = torch.from_numpy(input)
        input = input.to(device)
        with torch.no_grad():
            output = model(input.float())
        vector = output.cpu().squeeze().numpy()
        print(vector.shape)
        x_embedding.append(vector)
        print('======', i)
    x_embedding = np.array(x_embedding)
    pickle.dump(x_embedding, open(save_embedding, 'wb'), protocol=4)


if __name__ == '__main__':
    main()

