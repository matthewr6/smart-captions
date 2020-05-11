import random
import pandas as pd
import numpy as np
from glob import glob
import os

from tensorflow.keras.utils import to_categorical

from constants import VOCAB_SIZE, seqs_to_captions, nonrare_words

# TODO: ImageDataGenerator with flipping/rotating/etc?

# Note: longest caption is 25.  So, let's pad to 30.
# charwise, let's say 300
MAX_SEQ_LEN = 30
def load_captions():
    with open('../data/captions.txt') as f:
        data = f.readlines()
    loaded = {}
    for example in data:
        example = example[:-1].split(',')
        img_idx = example[0]
        word_vector = []
        for w_idx in example[1:]:
            if w_idx in nonrare_words:
                word_vector.append(int(w_idx) + 1)
        word_vector = np.pad(word_vector, (0, MAX_SEQ_LEN - len(word_vector)))
        loaded[img_idx] = to_categorical(word_vector, num_classes=VOCAB_SIZE)
    return loaded

# Read the image list and csv
image_list = glob('../data/processed/*.*')
data = load_captions()

cache = {}
use_cache = False
def load_image(image_path):
    if cache.get(image_path):
        return cache.get(image_path)
    image = np.load(image_path)
    if use_cache:
        cache[image_path] = image
    return image

def data_generator(batch_size=100):
    i = len(image_list)
    while True:
        # batch = {'images': [], 'captions': [], 'next_word': []}
        batch = {'images': [], 'captions': []}
        for b in range(batch_size):
            if i == len(image_list):
                i = 0
                random.shuffle(image_list)

            image_path = image_list[i]
            image_name = os.path.basename(image_path).split('.')[0]
            image = load_image(image_path)

            # Read data from csv using the name of current image
            # caption = df.loc[int(image_name)][0]
            caption = data[image_name]

            if image.shape[2] == 4:
                image = image[...,:3]

            # for i in range(MAX_SEQ_LEN):
            #     batch['images'].append(image)
            #     partial = caption[:i]
            #     padded_partial = np.pad(partial, ((0, MAX_SEQ_LEN - partial.shape[0]), (0, VOCAB_SIZE - partial.shape[1])))
            #     batch['captions'].append(padded_partial)
            #     batch['next_word'].append(caption[i])

            batch['images'].append(image)
            batch['captions'].append(caption)

            i += 1
            
        batch['images'] = np.array(batch['images'])
        batch['captions'] = np.array(batch['captions'])
        # batch['next_word'] = np.array(batch['next_word'])

        yield [batch['images'], batch['captions']]#, batch['next_word']]

def partial_generator(captions):
    batch = {'next_word': [], 'captions': []}
    for caption in captions:
        for i in range(MAX_SEQ_LEN):
            partial = caption[:i]
            padded_partial = np.pad(partial, ((0, MAX_SEQ_LEN - partial.shape[0]), (0, VOCAB_SIZE - partial.shape[1])))
            batch['captions'].append(padded_partial)
            batch['next_word'].append(caption[i])
    batch['captions'] = np.array(batch['captions'])
    batch['next_word'] = np.array(batch['next_word'])
    return batch['captions'], batch['next_word']

if __name__ == '__main__':
    gen = data_generator(batch_size=13)
    # for i in range(10):
    images, captions = next(gen)
    partial_captions, next_words = partial_generator(captions)
