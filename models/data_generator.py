import random
import pandas as pd
import numpy as np
from glob import glob
import os

from tensorflow.keras.utils import to_categorical

from constants import VOCAB_SIZE, seqs_to_captions, nonrare_words

# I think I need a start of sequence token as well as end of sequence (which is currently 0)...
# Also, let's separate end and null tokens; after offset of 2:
# start token = 0
# end token = 1
# null = all zeros
# word tokens start at 2

# Note: longest caption is 25.  So, let's pad to 30.
MAX_SEQ_LEN = 30
OFFSET = 2
# todo : my data is backwards! null tokens need to be at start
def load_captions():
    with open('../data/captions.txt') as f:
        data = f.readlines()
    loaded = {}
    for example in data:
        example = example[:-1].split(',')
        img_idx = example[0]
        word_vector = [0]
        for w_idx in example[1:]:
            if w_idx in nonrare_words:
                word_vector.append(int(w_idx) + 2)
        word_vector.append(1)
        # word_vector = np.pad(word_vector, (0, MAX_SEQ_LEN - len(word_vector)))
        word_vector = to_categorical(word_vector, num_classes=VOCAB_SIZE)
        loaded[img_idx] = np.pad(word_vector, ((MAX_SEQ_LEN - word_vector.shape[0], 0), (0, VOCAB_SIZE - word_vector.shape[1])))
    return loaded

# Read the image list and csv
# TEMP SLICING
image_list = glob('../data/processed/*.*')[:1]
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

def partial_generator(caption):
    batch = {'next_words': [], 'captions': []}
    # for caption in captions:
    for i in range(MAX_SEQ_LEN - 1, -1, -1):
        if np.argmax(caption[i]) == 0: # if we've encountered the start token, don't bother adding further examples
            break

        partial = caption[:i]
        # todo is this right
        # padded_partial = np.pad(partial, ((MAX_SEQ_LEN - partial.shape[0], 0), (0, VOCAB_SIZE - partial.shape[1])))
        padded_partial = np.pad(partial, ((MAX_SEQ_LEN - partial.shape[0], 0), (0, VOCAB_SIZE - partial.shape[1])))
        # padded_partial[:i, 2] = 1 
        batch['captions'].append(padded_partial)
        batch['next_words'].append(caption[i])
    batch['captions'] = np.array(batch['captions'])
    batch['next_words'] = np.array(batch['next_words'])
    return batch['captions'], batch['next_words']

if __name__ == '__main__':
    gen = data_generator(batch_size=13)
    # for i in range(10):
    images, captions = next(gen)
    print(np.argmax(captions[0], axis=1))
    partial_captions, next_words = partial_generator(captions[0])
    print(partial_captions.shape, next_words.shape)
    print(np.argmax(partial_captions, axis=2), np.argmax(next_words, axis=1))
