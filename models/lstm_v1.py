import numpy as np
import tensorflow as tf

import os
import csv
from collections import defaultdict as dd

from tensorflow.keras.preprocessing.text import Tokenizer


START = -3
NULL = -2
END = -1

def tokenize_captions(caption_path):
    images = []
    captions = []
    with open(caption_path) as f:
        r = csv.reader(f)
        for line in r:
            image, caption = line
            images.append(image)
            captions.append(caption)

    # remove the .jpg from images
    images = list(map(lambda image : image[:-4], images))

    # tokenize the captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    seqs = tokenizer.texts_to_sequences(captions)
    max_seq_len = max(map(len, seqs))

    # add the start / end tokens and pad all captions to the same length
    captions = dd(list)
    for image, seq in zip(images, seqs):
        if len(seq) < max_seq_len:
            seq += [NULL] * (max_seq_len - len(seq))
        seq = [START] + seq + [END]
        captions[image].append(seq)
    return tokenizer.index_word, captions

def get_split_generators(data, batch_size=100, split=(70, 20, 10)):
    np.random.shuffle(data)
    N = len(data)
    # get the cumulative percentages of the sutoffs for train and val
    train_p, val_p = split[0], split[0] + split[1]
    train, val, test = np.split(data, [N * train_p // 100, N * val_p // 100])

    generator_dict = {}
    generator_dict['train'] = define_data_generator(train, batch_size)
    generator_dict['val'] = define_data_generator(val, batch_size)
    generator_dict['test'] = define_data_generator(test, batch_size)

    return generator_dict

def load_image(image_path):
    image_path = '../data/processed_vgg16/{}.npy'.format(image_path)
    if not os.path.isfile(image_path):
        print('no image at', image_path)
        return None
    image = np.load(image_path)
    return image

def define_data_generator(data, batch_size=100):
    def data_gen():
        i = len(data)
        while True:
            batch = {'images': [], 'captions': [], 'next_words': []}
            for b in range(batch_size):
                if i == len(data):
                    i = 0
                    np.random.shuffle(data)
                
                # we have 5 captions because it is the flickr data
                image_name, captions = data[i]
                i += 1

                image = load_image(image_name)

                if image is None:
                    continue

                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = image[...,:3]

                # we will add the partials for every caption
                for caption in captions:
                    L = len(caption)
                    for j in range(1, L):
                        # we have reached the end of the caption
                        if caption[j] == NULL:
                            break
                        partial = caption[:j]
                        # our partial features is all previous words padded to our seq len, L
                        padded_partial = np.array([NULL] * (L - len(partial)) + partial)
                        batch['images'].append(image)
                        batch['captions'].append(padded_partial)
                        batch['next_words'].append(caption[j])

                
            batch['images'] = np.array(batch['images'])
            batch['captions'] = np.array(batch['captions'])
            batch['next_words'] = np.array(batch['next_words'])

            yield [batch['images'], batch['captions'], batch['next_words']]
    return data_gen


def main():
    idx_word, captions = tokenize_captions('../data/captions.csv')
    print('after tokenizing and condensing')
    data = list(captions.items())
    print(data[0])

    gen_dict = get_split_generators(data)
    train_gen = gen_dict['train']

    print('final data given to train with')
    batch = next(train_gen())
    images, captions, next_words = batch[0], batch[1], batch[2]
    print(len(images))
    print('image shape', images[55].shape, 'caption', captions[55], 'deindexed word', idx_word[next_words[55]])

    




if __name__ == '__main__':
    main()