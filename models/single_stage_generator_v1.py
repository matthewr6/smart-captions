import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Concatenate,
    LSTM,
    Conv2D,
    LeakyReLU,
    MaxPooling2D
)

OUTPUT_SIZE = 30

class SingleStageGeneratorV1():

    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.outputs = None
        self.model = None
        self.build()

    def build(self):
        img_input = Input(shape=(250, 250, 3))
        self.inputs = [img_input]

        pooling_freq = 2
        encoder = img_input
        for idx, filters in enumerate([64, 64, 128, 128, 128, 256, 1024]):
            encoder = Conv2D(filters, 3, activation='relu')(encoder)
            if (idx + 1) % pooling_freq == 0:
                encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

        encoder_output = Flatten()(encoder)

        # https://github.com/zhixuhao/unet/blob/master/model.py
        # todo - look more into how this works, RNN vs LSTM vs GRU
        # maybe just use dense layers directly for V1 as proof of concept...
        # decoder = LSTM(50)(encoder)
        # see section 10 of https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
        # i think the true output is the concat of sequence input + output (and the output we train on is the potentially the single item alone)
        # actually, I think the optimal solution is to have the generative network output a purely dense output, and have the adversarial network
        # take in an RNN input from this and predict 1/0 on that - see also https://towardsdatascience.com/generating-pokemon-inspired-music-from-neural-networks-bc240014132
        # i should also do skip connections (with one hidden layer in the skip) in v2!
        decoder = Dense(256)(encoder_output)
        for nodes in [256, 128, 128, 64, 64]:
            decoder = Dense(nodes)(decoder)
            decoder = LeakyReLU(alpha=0.2)(decoder)

        decoder_output = Dense(OUTPUT_SIZE, activation='sigmoid')(decoder)
        self.outputs = [decoder_output]

        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='single_stage_generator_v1')
    