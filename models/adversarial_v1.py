import numpy as np
import tensorflow as tf
from datetime import datetime

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

    # so many arbitrary design choices here rn.  very easy to modify.
    def build_discriminator(self):
        pooling_freq = 2

        # Image CNN
        img_input = Input(shape=(4096,), name='discrim_img_input')

        # Dense from CNN
        img = Dense(256, activation='relu')(img_input)
        img = Dropout(0.25)(img)

        # Caption sequence
        embed_dim = 256
        previous_words_input = Input(shape=(MAX_SEQ_LEN, ), name='discrim_caption_input')

        next_word_input = Input(shape=(VOCAB_SIZE, ), name='gen_prediction_input')
        next_word = Lambda(lambda x: backend.cast(backend.argmax(x, axis=1), 'float32'))(next_word_input)
        next_word = Reshape((1, ))(next_word)

        current_caption = Concatenate()([previous_words_input, next_word])
        recurrent_layer = Embedding(VOCAB_SIZE, embed_dim, mask_zero=True)(current_caption)

        combined = Add()([recurrent_layer, img])

        combined = LSTM(256)(combined)

        combined = Dense(128, activation='relu')(combined)
        img = Dropout(0.25)(img)

        combined = Dense(1, activation='sigmoid')(combined)

        self.discriminator = keras.Model(inputs=[img_input, previous_words_input, next_word_input], outputs=[combined], name='adversarial_output')

        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def connect_generator(self):
        # Ignore the warning; we want to make the discriminator trainable but only alone.

        img = Input(shape=(4096, ))
        partial_caption = Input(shape=(MAX_SEQ_LEN, ))

        generated_next_word = self.generator.model([img, partial_caption])

        self.discriminator.trainable = False
        valid = self.discriminator([img, partial_caption, generated_next_word])

        self.full_model = keras.Model(inputs=[img, partial_caption], outputs=[valid], name='combined')
        self.full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def show_model_structures(self):
        if self.discriminator:
            keras.utils.plot_model(self.discriminator, 'discriminator_{}.png'.format(self.name))
            print(self.discriminator.summary())
        if self.generator:
            keras.utils.plot_model(self.generator.model, 'generator_{}.png'.format(self.name))
            print(self.generator.model.summary())
        keras.utils.plot_model(self.full_model, 'full_{}.png'.format(self.name))
        print(self.full_model.summary())


    def train(self, generators, iters, batch_size=50):
        start_time = datetime.now()

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        train_generator = generators['train']()
        val_generator = generators['val']()
        for iteration in range(iters):
            images, captions, next_words = next(train_generator)
            next_words = utils.to_categorical(next_words, num_classes=VOCAB_SIZE)
            val_images, val_captions, val_next_words = next(val_generator)

            num_examples = len(images)
            real = np.ones((num_examples,))
            fake = np.zeros((num_examples,))

            fake_next_words = self.generator.model.predict([images, captions])
            real_loss = self.discriminator.train_on_batch([images, captions, next_words], real)
            fake_loss = self.discriminator.train_on_batch([images, captions, fake_next_words], fake)
            discriminator_stats = np.mean([real_loss, fake_loss], axis=0)

            generator_stats = self.full_model.train_on_batch([images, captions], real)

            elapsed_time = datetime.now() - start_time
            print('[Iter {}/{}] [D loss: {}, acc: {},{},mean{}] [Full loss: {}] [% realistic generated: {}] time: {}'.format(
                iteration,
                iters,
                round(discriminator_stats[0], 2),
                round(100 * real_loss[1], 2),
                round(100 * fake_loss[1], 2),
                round(100 * discriminator_stats[1], 2),
                # round(100 * generator_stats[4], 2),
                round(generator_stats[0], 2),
                round(100 * generator_stats[1], 2),
                elapsed_time
            ))

        self.full_model.save('model_{}'.format(self.name))

    def predict(self, images):
        return self.generator.predict(images)
