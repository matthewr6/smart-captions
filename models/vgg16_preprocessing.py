import glob
import os

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input 

vgg16 = VGG16(input_tensor=Input(shape=(250, 250, 3)))
vgg16_mod = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)

for path in glob.glob('../flickr8k_data/processed/*.npy'):
    image = np.load(path)
    basename = os.path.basename(path)
    result = vgg16_mod.predict(np.array([image]))[0]
    np.save('../flickr8k_data/processed_vgg16/{}'.format(basename), result)
