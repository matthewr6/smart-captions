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
    GaussianNoise
)

from constants import VOCAB_SIZE

INPUT_SIZE = 30
# INPUT_SIZE = 500
CHAR_SEQ_LEN = 500
NUM_CHARS = 29

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
        img_input = Input(shape=(250, 250, 3), name='discrim_img_input')
        pretrained_cnn = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)
        flexible_layers = 1 # arbitrary
        for layer in pretrained_cnn.layers[:-flexible_layers]:
            layer.trainable = False

        # Dense from CNN
        img = Flatten()(pretrained_cnn.output)
        img = Dense(1024, activation='relu')(img)
        img = Dropout(0.25)(img)
        img = Dense(256, activation='relu')(img)
        img = Dropout(0.25)(img)

        # Caption sequence
        caption_input = Input(shape=(INPUT_SIZE, VOCAB_SIZE), name='discrim_caption_input')
        latent_dim = 12 # arbitrary, inline in generative
        caption, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(caption_input)
        caption = Dense(24)(caption)
        caption = Dense(12)(caption)
        caption = Flatten()(caption)

        # Caption dense
        for nodes in [256, 64]:
            caption = Dense(nodes, activation='relu')(caption)
            caption = Dropout(0.25)(caption)

        # Combined dense
        combined = Concatenate(axis=-1)([img, caption])
        for nodes in [32, 32]: # arbitrary
            combined = Dense(nodes, activation='relu')(combined)
            combined = Dropout(0.25)(combined)

        combined = Dense(1, activation='sigmoid', name='discrim_output')(combined)

        self.discriminator = keras.Model(inputs=[img_input, caption_input], outputs=[combined], name='adversarial_caption_v1')
        self.discriminator_caption_input = caption_input
        # self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def connect_generator(self):
        # Ignore the warning; we want to make the discriminator trainable but only alone.
        self.discriminator.trainable = False
        adversarial_output = self.discriminator(self.generator.inputs + self.generator.outputs)

        # potentially train full model with generator output also, but that might not be necessary.
        # self.full_model = keras.Model(inputs=self.generator.inputs, outputs=self.generator.outputs + [adversarial_output], name='combined')
        self.full_model = keras.Model(inputs=self.generator.inputs, outputs=adversarial_output, name='combined')
        optimizer = Adam(lr=0.00001)
        self.full_model.compile(
            optimizer=optimizer,
            loss={
                'adversarial_caption_v1': 'binary_crossentropy',
                # 'gen_output': 'categorical_crossentropy', # unsure if this is good, or whether to use it
            },
            metrics=['accuracy']
        )

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
            fake_captions = self.generator.model.predict(images)
            real_loss = self.discriminator.train_on_batch([images, real_captions], real)
            fake_loss = self.discriminator.train_on_batch([images, fake_captions], fake)
            discriminator_stats = np.mean([real_loss, fake_loss], axis=0)

            generator_stats = self.full_model.train_on_batch(images, real)
            # generator_stats = self.full_model.train_on_batch(images, [real_captions, real])

            elapsed_time = datetime.now() - start_time
            print('[Epoch {}/{}] [D loss: {}, acc: {},{},mean{}] [Full loss: {}] [% realistic generated: {}] time: {}'.format(
                epoch,
                epochs,
                round(discriminator_stats[0], 2),
                round(100 * real_loss[1], 2),
                round(100 * fake_loss[1], 2),
                round(100 * discriminator_stats[1], 2),
                # round(100 * generator_stats[4], 2),
                round(generator_stats[0], 2),
                round(100 * generator_stats[1], 2),
                elapsed_time
            ))

            if epoch >= epochs:
                break

        self.full_model.save('model_{}'.format(self.name))

    def predict(self, images):
        return self.generator.predict(images)
