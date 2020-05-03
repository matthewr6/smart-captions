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
    LeakyReLU
)

# Call model = AdversarialModel(GeneratorModelName(), generator)
class AdversarialModel():

    def __init__(self, generator, name='experimental', *args, **kwargs):
        self.generator = generator
        self.data = data
        self.name = name
        self.discriminator = None
        self.full_model = None
        self.build_discriminator()
        self.connect_generator()
        self.show_model_structures()

    # so many arbitrary design choices here rn.  very easy to modify.
    def build_discriminator(self):
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

        self.discriminator = keras.Model(inputs=[img_input, caption_input], outputs=[combined], name='adversarial_caption')

    def connect_generator(self):
        # i have no idea if the inputs/outputs works but this is the genearl idea.
        self.full_model = keras.Model(inputs=self.generator.inputs, outputs=self.generator.outputs + self.discriminator.outputs)

    def show_model_structures(self):
        if self.discriminator:
            keras.utils.plot_model(self.discriminator, 'discriminator_{}.png'.format(self.name))
            print(self.discriminator.summary())
        if self.generator:
            keras.utils.plot_model(self.generator, 'generator_{}.png'.format(self.name))
            print(self.generator.summary())

    def train(self, epochs, batch_size=10):
        real = np.ones((batch_size,))
        fake = np.zeros((batch_size,))
        for epoch in range(epochs):
            # design choice - this iterates thru N individually per epoch; do we want to
            # not iterate and just batch? let's find out thru speed tests...
            for batch_idx, (images, real_captions) in enumerate(self.data(batch_size=batch_size)):
                fake_captions = self.generator.predict(images)

                real_loss = self.discriminator.train_on_batch([images, real_captions], real)
                fake_loss = self.discriminator.train_on_batch([images, fake_captions], fake)

                discriminator_loss = np.mean([real_loss, fake_loss], axis=0)

                # this also makes assumptions; i assume they're justified in the pix2pix paper.
                # we can also freeze the discriminator model's weights before training the combined model.
                # https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
                generator_loss = self.full_model.train_on_batch([images, real_captions], [real_captions, valid])

                print('Epoch {}/{}:\n\tD Loss {}, acc {}\n\tG Loss {}'.format(epoch, epochs, discriminator_loss, ))
