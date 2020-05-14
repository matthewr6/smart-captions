import tensorflow as tf
import sys, os
sys.path.append('../preprocessing')
from reduce_captions import get_captions, get_lookup

import random
from sklearn.model_selection import train_test_split
import numpy as np

cache = {}
use_cache = False
def load_image(image_path):
    if cache.get(image_path):
        return cache.get(image_path)
    image_path = '../data/processed/{}.npy'.format(image_path)
    print(image_path)
    if not os.path.isfile(image_path):
        return None
    print('load')
    image = np.load(image_path)
    if use_cache:
        cache[image_path] = image
    return image

def get_data(labels):
    X = []
    y = []
    for i in range(23376):
        try:
            arr = np.load('../data/processed/' + str(i) + '.npy')
        except FileNotFoundError:
            continue
        print(arr.shape, i)
        X.append(arr)
        y.append(labels[i])
    
    X = np.array(X)
    y = np.array(y)
    return (X, y)

def load_transfer_data():
    nouns = []
    with open('../data/caption_nouns.txt', 'r') as f:
        for line in f:
            nouns.append(line[:-1])
    words = get_lookup('nostem')
    captions = get_captions('nostem')
    labels = {}
    for i, caption in captions.items():
        word_indexes = captions[i]
        label = np.zeros(len(nouns))
        for word_index in word_indexes:
            word = words[word_index]
            if word in nouns:
                label[nouns.index(word)] = 1
        labels[i] = label
    data = []
    for i, caption in labels.items():
        data.append((i, caption))
    return np.array(data)

def transfer_generator(batch_size=100):
    data = load_transfer_data()
    i = len(data)
    while True:
        batch = {'images': [], 'captions': []}
        for b in range(batch_size):
            if i == len(data):
                i = 0
                random.shuffle(data)

            image_name, caption = data[i]
            i += 1

            image = load_image(str(image_name))

            if image is None:
                continue

            batch['images'].append(image)
            batch['captions'].append(caption)

            
        batch['images'] = np.array(batch['images'])
        batch['captions'] = np.array(batch['captions'])

        yield [batch['images'], batch['captions']]


def test_model(verbose=False):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='sigmoid'))
    if verbose:
        model.summary()

def main():
    # print('started')
    # gen = transfer_generator(batch_size=13)
    # images, captions = next(gen)
    # print(images.shape, captions)
    # labels = get_labels()
    # X, y = get_data(labels)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27, test_size=0.1)
    # test_model(verbose=True)



if __name__ == '__main__':
    main()