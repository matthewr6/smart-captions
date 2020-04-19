import random
import pandas as pd
import numpy as np
from glob import glob
import os
# from keras.preprocessing import image as krs_image

# Create an empty data generator
# datagen = ImageDataGenerator()

# Read the image list and csv
image_list = glob('data/processed/*.*')
df = pd.read_csv('data/captions.csv', header=None, names=['id', 'caption'])
df.set_index('id', inplace=True)

img_height = 400
img_width = 400

def generator(num=100):
    i = len(image_list)
    while True:
        batch = {'images': [], 'captions': []}
        for b in range(num):
            if i == len(image_list):
                i = 0
                # random.shuffle(image_list)

            image_path = image_list[i]
            image_name = os.path.basename(image_path).split('.')[0]
            image = np.load(image_path)

            # Read data from csv using the name of current image
            caption = df.loc[int(image_name)][0]

            batch['images'].append(image)
            # print(image.shape)
            batch['captions'].append(caption)

            i += 1

        batch['images'] = np.array(batch['images'])
        batch['captions'] = np.array(batch['captions'])

        yield [batch['images'], batch['captions']]

gen = generator(num=100)
n = next(gen)
print(n[0].shape, n[1].shape)