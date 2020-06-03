import random
import pandas as pd
import numpy as np
from glob import glob
import os

from constants import VOCAB_SIZE, seqs_to_captions, nonrare_words, MAX_SEQ_LEN, NUM_SIGNAL_TOKENS

# I think I need a start of sequence token as well as end of sequence (which is currently 0)...
# Also, let's separate end and null tokens; after offset of 2:
# start token = 1
# end token = 2
# null = 0
# word tokens start at 3

def load_captions(filename):
    with open(filename) as f:
        data = f.readlines()
    loaded = []
    for example in data:
        example = example[:-1].split(',')
        img_name = example[0].split('.')[0] # the first value in each csv row is the img_name
        word_vector = [1]
        for w_idx in example[1:]: # get the list of words in the example
            if w_idx in nonrare_words:
                word_vector.append(int(w_idx) + NUM_SIGNAL_TOKENS)
        word_vector.append(2)
        # create an example for every partial caption (any subsequence of the caption)
        loaded.append((
            img_name,
            np.pad(word_vector, ((MAX_SEQ_LEN - len(word_vector), 0)))
        ))
    return loaded

# Read the image list and csv
# all_data = load_captions() # array of (name, encoded caption)

cache = {}
use_cache = False
def load_image(image_path, mode='vgg16'):
    if cache.get(image_path):
        return cache.get(image_path)
    if mode == 'vgg16':
        image_path = 'C:/Users/FitzL/Desktop/data/processed_vgg16/{}.npy'.format(image_path)
    else:
        image_path = '../data/processed/{}.npy'.format(image_path)
    if not os.path.isfile(image_path):
        return None
    image = np.load(image_path)
    if use_cache:
        cache[image_path] = image
    return image

def get_split_generators(all_data, batch_size=100, mode='vgg16', split=(70, 20, 10)):
    random.shuffle(all_data)
    N = len(all_data)
    # get the cumulative percentages of the sutoffs for train and val
    train_p, val_p = split[0], split[0] + split[1]
    train, val, test = np.split(all_data, [N * train_p // 100, N * val_p // 100])

    generator_dict = {}
    generator_dict['train'] = define_data_generator(train, batch_size, mode)
    generator_dict['val'] = define_data_generator(val, batch_size, mode)
    generator_dict['test'] = define_data_generator(test, batch_size, mode)

    print("--------------------------------------")
    return generator_dict



def define_data_generator(data, batch_size=100, mode='vgg16'):
    def data_gen():
        i = len(data)
        while True:
            batch = {'images': [], 'captions': [], 'next_words': []}
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    random.shuffle(data)
                image_name, caption = data[i]
                i += 1

                image = load_image(image_name, mode=mode)

                if image is None:
                    continue

                if len(image.shape) == 3 and image.shape[2] == 4 and mode != 'vgg16':
                    image = image[...,:3]

                for j in range(MAX_SEQ_LEN - 1, -1, -1):
                    if caption[j] == 1: # if we've encountered the start token, don't bother adding further examples
                        break

                    partial = caption[:j]
                    padded_partial = np.pad(partial, (MAX_SEQ_LEN - partial.shape[0], 0))
                    batch['images'].append(image)
                    batch['captions'].append(padded_partial)
                    batch['next_words'].append(caption[j])

                
            batch['images'] = np.array(batch['images'])
            batch['captions'] = np.array(batch['captions'])
            batch['next_words'] = np.array(batch['next_words'])

<<<<<<< HEAD
            return [batch['images'], batch['captions'], batch['next_words']]
    return data_gen
=======
            yield [batch['images'], batch['captions'], batch['next_words']]
    return data_gen 
>>>>>>> e12de3ad2a01c6b763a54cbdb8d095cec8db2804

if __name__ == '__main__':

    # gen = data_generator(batch_size=13)
    # images, captions, next_words = next(gen)
    # print(images.shape, captions.shape, next_words.shape)

    # Usage example!
    filename = ".../data/captions.txt"
    data = load_captions(filename)
    gen_dict = get_split_generators(data, batch_size=5)
    train = gen_dict['train']
    print(next(train()))
