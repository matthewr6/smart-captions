import numpy as np
import tensorflow as tf
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import (
    SGD,
    Adam,
    Adamax,
)
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
)
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Concatenate,
    LSTM,
    Conv2D,
    LeakyReLU,
    MaxPooling2D,
    Dropout,
    GaussianNoise,
    Embedding,
    Add,
    Lambda,
)

from tensorflow.keras import backend, utils

from constants import VOCAB_SIZE, MAX_SEQ_LEN

# Call model = AdversarialModel(GeneratorModelName(), generator)
class AdversarialModelV1():

    def __init__(self, generator, name='experimental', *args, **kwargs):
        self.generator = generator
        # self.data = data
        self.name = name
        self.discriminator = None
        self.full_model = None
        self.build_discriminator()
        self.connect_generator()
        self.show_model_structures()

    def build_discriminator(self):
        previous_words_input = Input(shape=(MAX_SEQ_LEN, ), name='discrim_caption_input')        
        next_word_input = Input(shape=(VOCAB_SIZE, ), name='gen_prediction_input')
        next_word = Dense(1, activation='relu')(next_word_input)

        current_caption = Concatenate()([previous_words_input, next_word])
        recurrent_layer = Embedding(VOCAB_SIZE, 256, mask_zero=True)(current_caption)
        
        img_input = Input(shape=(4096,), name='discrim_img_input')
        img = Dense(256, activation='relu')(img_input)
        img = Dropout(0.25)(img)

        combined = Add()([recurrent_layer, img])

        combined = LSTM(256)(combined)

        combined = Dense(128, activation='relu')(current_caption)
        final = Dense(1, activation='sigmoid')(combined)

        self.discriminator = keras.Model(inputs=[img_input, previous_words_input, next_word_input], outputs=[final], name='adversarial_output')
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def connect_generator(self):
        img = Input(shape=(4096, ))
        partial_caption = Input(shape=(MAX_SEQ_LEN, ))

        generated_next_word = self.generator.model([img, partial_caption])

        # Ignore the warning; we want to make the discriminator trainable but only alone.
        self.discriminator.trainable = False
        valid = self.discriminator([img, partial_caption, generated_next_word])

        self.full_model = keras.Model(inputs=[img, partial_caption], outputs=[valid], name='combined')
        self.full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def show_model_structures(self):
        if self.discriminator:
            keras.utils.plot_model(self.discriminator, 'discriminator_{}.png'.format(self.name))
            # print(self.discriminator.summary())
        if self.generator:
            keras.utils.plot_model(self.generator.model, 'generator_{}.png'.format(self.name))
            # print(self.generator.model.summary())
        keras.utils.plot_model(self.full_model, 'full_{}.png'.format(self.name))
        # print(self.full_model.summary())


    def train(self, generators, iters, batch_size=50):
        start_time = datetime.now()

        history = {
            'd_real_acc': [],
            'd_fake_acc': [],
            'realistic_generated': [],

            'd_val_real_acc': [],
            'd_val_fake_acc': [],
            'val_realistic_generated': [],
        }

        train_generator = generators['train']()
        val_generator = generators['val']()
        for iteration in range(iters):
            images, captions, next_words = next(train_generator)
            next_words = utils.to_categorical(next_words, num_classes=VOCAB_SIZE)
            val_images, val_captions, val_next_words = next(val_generator)
            val_next_words = utils.to_categorical(val_next_words, num_classes=VOCAB_SIZE)

            num_examples = len(images)
            num_val_examples = len(val_images)
            real = np.ones((num_examples,))
            fake = np.zeros((num_examples,))
            val_real = np.ones((num_val_examples,))
            val_fake = np.zeros((num_val_examples,))

            fake_next_words = self.generator.model.predict([images, captions])
            _, real_acc = self.discriminator.train_on_batch([images, captions, next_words], real)
            _, fake_acc = self.discriminator.train_on_batch([images, captions, fake_next_words], fake)
            _, generated_acc = self.full_model.train_on_batch([images, captions], real)

            val_fake_next_words = self.generator.model.predict([val_images, val_captions])
            _, real_val_acc = self.discriminator.test_on_batch([val_images, val_captions, val_next_words], val_real)
            _, fake_val_acc = self.discriminator.test_on_batch([val_images, val_captions, val_fake_next_words], val_fake)
            _, val_generated_acc = self.full_model.test_on_batch([val_images, val_captions], val_real)

            elapsed_time = datetime.now() - start_time
            print('[Iter {}/{}]\n\t[D acc (real/fake): {}, {}]\n\t[Realistic generated: {}]\n\t[Val D acc (real/fake): {}, {}]\n\t[Val realistic: {}]\n\t[Time: {}]'.format(
                iteration,
                iters,
                real_acc,
                fake_acc,
                generated_acc,
                real_val_acc,
                fake_val_acc,
                val_generated_acc,
                elapsed_time
            ))

            history['d_real_acc'].append(real_acc)
            history['d_fake_acc'].append(fake_acc)
            history['realistic_generated'].append(generated_acc)

            history['d_val_real_acc'].append(real_val_acc)
            history['d_val_fake_acc'].append(fake_val_acc)
            history['val_realistic_generated'].append(val_generated_acc)

        for name, values in history.items():
            print(name)
            plt.plot(range(0, iters), values, label=name)
            plt.xlabel('Iteration')
            plt.ylabel(name)
            plt.legend()
            plt.savefig('{}_history.png'.format(name))
            plt.clf()

        self.full_model.save('model_{}'.format(self.name))

    def predict(self, images):
        return self.generator.predict(images)

    def single_predict(self, images, captions):
        return self.generator.single_predict(images, captions)
