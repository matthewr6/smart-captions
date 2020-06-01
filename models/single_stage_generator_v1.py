import numpy as np
import tensorflow as tf

from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
)
from tensorflow.keras.optimizers import (
    RMSprop,
    Adagrad,
    Adadelta,
    Adam
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
    RepeatVector,
    GRU,
    SimpleRNN,
    BatchNormalization,
    TimeDistributed,
    Embedding,
    Add
)

from tensorflow.keras.losses import (
    CategoricalCrossentropy
)

from tensorflow.keras.utils import to_categorical

from constants import VOCAB_SIZE, MAX_SEQ_LEN

class SingleStageGeneratorV1():

    def __init__(self, *args, **kwargs):
        self.model = None
        self.compiled = False
        self.build()

    def build(self):
        img_input = Input(shape=(4096,), name='vgg16_processed_input')
        img = Dense(256, activation='relu')(img_input)
        img = Dropout(0.25)(img)

        embed_dim = 256
        previous_words = Input(shape=(MAX_SEQ_LEN, ), name='gen_cur_words')
        recurrent_layer = Embedding(VOCAB_SIZE, embed_dim, mask_zero=True)(previous_words)

        combined = Add()([recurrent_layer, img])
        
        combined = LSTM(256)(combined)

        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.25)(combined)
        
        predicted_word = Dense(VOCAB_SIZE, activation='softmax')(combined)

        self.model = keras.Model(inputs=[img_input, previous_words], outputs=[predicted_word], name='single_stage_generator_v1')

    def compile(self):
        # rmsprop, adagrad, adadelta might be best options
        # optimizer = Adagrad(lr=0.01)
        # optimizer = RMSprop()
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        self.compiled = True
        print(self.model.summary())

    def prepare_new_sequence(self, copies=1):
        return np.zeros((copies, MAX_SEQ_LEN))

    def single_predict(self, images, sequences):
        return self.model.predict([images, sequences])

    def sample(self, a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1)) 

    def predict(self, images):
        captions = []
        for image in images:
            initial = np.zeros((MAX_SEQ_LEN, ))
            initial[-1] = 1

            for i in range(MAX_SEQ_LEN - 1, 0, -1):
                prediction = self.model.predict([np.array([image]), np.array([initial])])[0]
                initial = np.roll(initial, -1, axis=0)
                # initial[-1] = self.sample(prediction)
                initial[-1] = np.argmax(prediction)
                if np.argmax(prediction) == 2:
                    break
            captions.append(initial)
        return captions

    def load(self, load_path):
        self.model = keras.models.load_model(load_path)

    def train(self, generators, iters, batch_size=10):
        if not self.compiled:
            self.compile()
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
            val_images, val_captions, val_next_words = next(val_generator)

            loss, accuracy = self.model.train_on_batch([images, captions], next_words)
            val_loss, val_accuracy = self.model.test_on_batch([val_images, val_captions], val_next_words)
            elapsed_time = datetime.now() - start_time

            history['train_loss'].append(loss)
            history['train_acc'].append(accuracy)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            print('[Iter {}/{}] [Loss: {}] [Acc%: {}] [Val loss: {}] [Val acc%: {}] time: {}'.format(
                iteration,
                iters,
                loss,
                100 * accuracy,
                val_loss,
                100 * val_accuracy,
                elapsed_time,
            ))

        for name, values in history.items():
            print(name)
            plt.plot(range(0, iters), values, label=name)
            plt.xlabel('Epochs')
            plt.ylabel(name)
            plt.legend()
            plt.savefig('{}_history.png'.format(name))
            plt.clf()

        self.model.save('model_generator')

if __name__ == '__main__':
    model = SingleStageGeneratorV1()
    model.compile()
