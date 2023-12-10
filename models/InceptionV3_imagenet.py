import numpy as np
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = InceptionV3(weights='imagenet')

img_path = '/raid/yinghua/PriCod/data/imagenet/train/n09421951/n09421951_4884.JPEG'

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds_vec = model.predict(x)            # 一千维度的向量[1.43228512e-10 1.95161443e-09 3.06736303e-10 ...]
pre = decode_predictions(preds_vec, top=1) # [[('n09421951', 'sandbar', 0.9377819)]]
y = pre[0][0][0]
probability = pre[0][0][2]
print(y)
print(probability)
