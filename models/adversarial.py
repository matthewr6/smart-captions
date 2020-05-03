import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, LSTM, Conv2D, LeakyReLU


def discriminator():
    img_input = Input(shape=(250, 250, 3))
    for filters in [32, 32, 64, 64, 32, 32, 16]: # pretty arbitrary right now
        img = Conv2D(filters, kernel_size=3, strides=2, padding='same')(img_input)
        img = LeakyReLU(alpha=0.2)(img)
    img = Flatten()(img)
    for nodes in [256, 128, 64]:
        img = Dense(nodes)(img)
        img = LeakyReLU(alpha=0.2)(img)

    caption_input = Input(shape=(None, 50)) # todo - need to pad input
    caption = LSTM(25)(caption_input)
    for nodes in [32, 64, 64]:
        caption = Dense(nodes)(caption)
        caption = LeakyReLU(alpha=0.2)(caption)

    combined = Concatenate(axis=-1)([img, caption])
    for nodes in [32]: # arbitrary
        combined = Dense(32)(combined)
        combined = LeakyReLU(alpha=0.2)(combined)
    combined = Dense(1, activation='sigmoid')(combined)

    model = keras.Model(inputs=[img_input, caption_input], outputs=[combined], name='adversarial_caption')
    keras.utils.plot_model(model, 'adversarial_caption.png')
    print(model.summary())
    return model

discriminator()
