import tensorflow as tf
import sys, os
sys.path.append('../preprocessing')
from reduce_captions_v2 import get_captions, get_lookup
from matplotlib import pyplot as plt


from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import (
    VGG16,
    VGG19
)

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    MaxPooling2D,
    Dropout,
    Conv2D,
    Dense,
    Flatten,
    Reshape
)

from tensorflow.keras.optimizers import (
    Adam
)

import random
from sklearn.model_selection import train_test_split
import numpy as np

cache = {}
use_cache = False
def load_image(image_path):
    if cache.get(image_path):
        return cache.get(image_path)
    image_path = 'C:/Users/FitzL/Desktop/data/processed_vgg16/{}.npy'.format(image_path)
    if not os.path.isfile(image_path):
        return None
    image = np.load(image_path)
    if use_cache:
        cache[image_path] = image
    return image

def load_transfer_data(filename):
    nouns = []
    with open('C:/Users/FitzL/Desktop/data/caption_nouns.txt', 'r') as f:
        for line in f:
            nouns.append(line[:-1])
    words = get_lookup()
    captions = get_captions(filename)
    labels = {}
    for i, caption in captions.items():
        word_indexes = captions[i]
        label = np.zeros(len(nouns))
        for word_index in word_indexes:
            word = words[int(word_index)]
            if word in nouns:
                label[nouns.index(word)] = 1
        labels[i] = label
    data = []
    for i, caption in labels.items():
        data.append((i, caption))
    return np.array(data)

def transfer_generator(filename, batch_size=100):
    data = load_transfer_data(filename)
    i = len(data)
    while True:
        batch = {'images': [], 'captions': []}
        for b in range(batch_size):
            if i == len(data):
                i = 0
                random.shuffle(data)

            image_name, caption = data[i]
            i += 1

            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_name = image_name[:-len(".jpg")]

            image = load_image(image_name)
            if image is None:
                continue
            elif image.shape != (4096,):
                print('\n', image_name, 'gives shape', image.shape)
                continue

            batch['images'].append(image)
            batch['captions'].append(caption)

        batch['images'] = np.array(batch['images'])
        batch['captions'] = np.array(batch['captions'])
        yield (batch['images'], batch['captions'])



def get_basic_model(verbose=False):
    model = Sequential()
    model.add(Reshape((64,64,1), input_shape=(4096,)))

    model.add(Conv2D(filters=16, kernel_size=(1, 1), activation="relu", input_shape=(64,64,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(65, activation='sigmoid'))
    
    if verbose:
        model.summary()
    return model

def decode_binarized_caption(caption, filename, ext=""):
    nouns = []
    with open('C:/Users/FitzL/Desktop/data/caption_nouns.txt', 'r') as f:
        for line in f:
            nouns.append(line[:-1])
    out = set()
    for i in range(len(caption)):
        if caption[i] > 0.5:
            out.add(nouns[i])
    return out

def main():
    model = get_basic_model(verbose=False)
    opt = Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt)
    
    preds = []
    losses = []

    for i in range(1, 6):
        filename = "C:/Users/FitzL/Desktop/data/captions_" + str(i) + ".txt"
        print("Current file: " + filename)
        batch = next(transfer_generator(filename))
        print(decode_binarized_caption(batch[1][0], filename))

        H = model.fit(transfer_generator(filename), epochs=15, steps_per_epoch=5)

        pred = np.round(model.predict(batch[0]))
        preds.append(pred)

        print(pred)
        print(batch[1])
        for i in range(10):
            print(decode_binarized_caption(pred[i], filename), decode_binarized_caption(batch[1][i], filename))
        
        losses.append(H.history["loss"])
    
    for i in range(5):
        print("Loss for captions_" + str(i+1))
        print(losses[i])



if __name__ == '__main__':
    main()