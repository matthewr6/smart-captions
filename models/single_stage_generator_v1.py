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

from data_generator import partial_generator

RecurrentLayer = GRU

# NUM_CHARS = 29

from constants import VOCAB_SIZE, MAX_SEQ_LEN

class SingleStageGeneratorV1():

    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.outputs = None
        self.model = None
        self.compiled = False
        self.build()

    def build(self):
        img_input = Input(shape=(4096,), name='vgg16_processed_input')
        img = Dense(128, activation='relu')(img_input)
        img = Dense(256, activation='relu')(img_input)
        # img = Dropout(0.3)(img)

        embed_dim = 256
        previous_words = Input(shape=(MAX_SEQ_LEN, ), name='gen_cur_words')
        recurrent_layer = Embedding(VOCAB_SIZE, embed_dim, mask_zero=True)(previous_words)
        # recurrent_layer = Dropout(0.3)(recurrent_layer)
        
        # recurrent_layer = LSTM(256)(recurrent_layer)

        combined = Add()([recurrent_layer, img])
        
        combined = LSTM(256)(combined)

        combined = Dense(256, activation='relu')(combined)
        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        # combined = Dropout(0.3)(combined)
        combined = Dense(VOCAB_SIZE, activation='softmax')(combined)

        self.inputs = [img_input, previous_words]
        self.outputs = [combined]
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='single_stage_generator_v1')

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

    def train(self, datagen, epochs, batch_size=10):
        if not self.compiled:
            self.compile()
        start_time = datetime.now()
        history = []
        for epoch, (images, captions, next_words) in enumerate(datagen(batch_size=batch_size)):
            loss, accuracy = self.model.train_on_batch([images, captions], next_words)
            elapsed_time = datetime.now() - start_time
            history.append((loss, accuracy))
            print('[Epoch {}/{}] [Loss: {}] [Acc%: {}] time: {}'.format(
                epoch,
                epochs,
                loss,
                100 * accuracy,
                elapsed_time,
            ))

            if epoch >= epochs:
                break

        plt.plot(range(0, epochs + 1), history[0], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_history.png')
        plt.clf()
        plt.plot(range(0, epochs + 1), history[1], label='Training Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig('acc_history.png')

        self.model.save('model_generator')

if __name__ == '__main__':
    model = SingleStageGeneratorV1()
    model.compile()
