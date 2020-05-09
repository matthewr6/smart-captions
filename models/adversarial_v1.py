import numpy as np
import tensorflow as tf
from datetime import datetime

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
    LeakyReLU
)

INPUT_SIZE = 30

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
        img_input = Input(shape=(250, 250, 3))
        for filters in [32, 32, 32, 32, 32, 32, 32, 32, 32]: # pretty arbitrary right now
            img = Conv2D(filters, kernel_size=3, strides=2, padding='same')(img_input)
            img = LeakyReLU(alpha=0.2)(img)
        img = Flatten()(img)
        for nodes in [256, 128, 64]:
            img = Dense(nodes)(img)
            img = LeakyReLU(alpha=0.2)(img)

        caption_input = Input(shape=(INPUT_SIZE,)) # todo - need to pad input automatically in model training? or preprocessing?
        # todo - look more into how this works, RNN vs LSTM vs GRU
        # caption = LSTM(25)(caption_input)
        # for V1 let's just go simple.
        caption = Dense(64)(caption_input)
        for nodes in [128, 256, 128]:
            caption = Dense(nodes)(caption)
            caption = LeakyReLU(alpha=0.2)(caption)

        combined = Concatenate(axis=-1)([img, caption])
        for nodes in [32]: # arbitrary
            combined = Dense(nodes)(combined)
            combined = LeakyReLU(alpha=0.2)(combined)
        combined = Dense(1, activation='sigmoid')(combined)

        self.discriminator = keras.Model(inputs=[img_input, caption_input], outputs=[combined], name='adversarial_caption_v1')
        self.discriminator_caption_input = caption_input
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def connect_generator(self):
        self.discriminator.trainable = False
        # Ignore the warning; we want to make the discriminator trainable but only alone.
        adversarial_output = self.discriminator(self.generator.inputs + self.generator.outputs)
        self.full_model = keras.Model(inputs=self.generator.inputs, outputs=self.generator.outputs + [adversarial_output], name='combined')
        # self.full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # TODO LATER: change to this
        self.full_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def show_model_structures(self):
        if self.discriminator:
            keras.utils.plot_model(self.discriminator, 'discriminator_{}.png'.format(self.name))
            print(self.discriminator.summary())
        if self.generator:
            keras.utils.plot_model(self.generator.model, 'generator_{}.png'.format(self.name))
            print(self.generator.model.summary())
        keras.utils.plot_model(self.full_model, 'full_{}.png'.format(self.name))
        print(self.full_model.summary())


    def train(self, datagen, epochs, batch_size=50):
        real = np.ones((batch_size,))
        fake = np.zeros((batch_size,))
        start_time = datetime.now()
        for epoch, (images, real_captions) in enumerate(datagen(batch_size=batch_size)):
            # design choice - this iterates thru N individually per epoch; do we want to
            # not iterate and just batch? let's find out thru speed tests...
            # for batch_idx, (images, real_captions) in enumerate(datagen(batch_size=batch_size)):

            fake_captions = self.generator.model.predict(images)

            real_loss = self.discriminator.train_on_batch([images, real_captions], real)
            fake_loss = self.discriminator.train_on_batch([images, fake_captions], fake)

            discriminator_stats = np.mean([real_loss, fake_loss], axis=0)

            # this also makes assumptions; i assume they're justified in the pix2pix paper.
            # we can also freeze the discriminator model's weights before training the combined model.
            # https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

            generator_stats = self.full_model.train_on_batch([images, real_captions], [real_captions, real])

            elapsed_time = datetime.now() - start_time
            print ('[Epoch {}/{}] [D loss: {}, acc: {}] [G loss: {}] time: {}'.format(epoch, epochs, round(discriminator_stats[0], 2), round(100*discriminator_stats[1], 2), round(generator_stats[0], 2), elapsed_time))
            # print ('[Epoch {}/{}] [D loss: {}, acc: {}] time: {}'.format(epoch, epochs, discriminator_stats[0], 100*discriminator_stats[1], elapsed_time))

            if epoch >= epochs:
                break
