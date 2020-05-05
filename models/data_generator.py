import random
import pandas as pd
import numpy as np
from glob import glob
import os

# TODO: ImageDataGenerator with flipping/rotating/etc?

# Note: longest caption is 25.  So, let's pad to 30.
PAD_SIZE = 30
EOM = -1
def load_captions():
    with open('../data/captions.txt') as f:
        data = f.readlines()
    loaded = {}
    for example in data:
        example = example[:-1].split(',')
        img_idx = example[0]
        word_vector = [int(w) for w in example[1:]]
        word_vector.append(EOM)
        loaded[img_idx] = np.pad(word_vector, (0, PAD_SIZE - len(word_vector)))
    return loaded

# Read the image list and csv
image_list = glob('../data/processed/*.*')
data = load_captions()

img_height = 400
img_width = 400

def data_generator(batch_size=100):
    i = len(image_list)
    while True:
        batch = {'images': [], 'captions': []}
        for b in range(batch_size):
            if i == len(image_list):
                i = 0
                # random.shuffle(image_list)

            image_path = image_list[i]
            image_name = os.path.basename(image_path).split('.')[0]
            image = np.load(image_path)

            # Read data from csv using the name of current image
            # caption = df.loc[int(image_name)][0]
            caption = data[image_name]

            batch['images'].append(image)
            # print(image.shape)
            batch['captions'].append(caption)

            i += 1

        batch['images'] = np.array(batch['images'])
        batch['captions'] = np.array(batch['captions'])

        yield [batch['images'], batch['captions']]

if __name__ == '__main__':
    gen = data_generator(batch_size=1000)
    for i in range(10):
        n = next(gen)
        print(n[0].shape, n[1].shape)