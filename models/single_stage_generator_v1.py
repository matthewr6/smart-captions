import numpy as np
import tensorflow as tf

from datetime import datetime

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
    Adadelta
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
    TimeDistributed
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
        img_input = Input(shape=(250, 250, 3), name='gen_img_input')
        self.inputs = [img_input]

        pretrained_cnn = VGG16(weights='imagenet', include_top=True, input_tensor=img_input)
        for layer in pretrained_cnn.layers:
            layer.trainable = False

        # img = BatchNormalization()(pretrained_cnn.output)
        img = Dense(256, activation='relu')(pretrained_cnn.output)
        img = RepeatVector(MAX_SEQ_LEN)(img)


        # https://github.com/chen0040/keras-image-captioning/blob/master/keras_image_captioning/library/vgg16_lstm.py

        previous_words = Input(shape=(MAX_SEQ_LEN, VOCAB_SIZE), name='gen_cur_words')
        recurrent_layer = LSTM(256, return_sequences=True)(previous_words)

        combined = Concatenate()([img, recurrent_layer])
        combined = LSTM(1024, return_sequences=False)(combined)
        combined = Dense(VOCAB_SIZE, activation='softmax')(combined)

        self.inputs.append(previous_words)
        self.outputs = [combined]
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='single_stage_generator_v1')

    def compile(self):
        # rmsprop, adagrad, adadelta might be best options
        optimizer = Adagrad(lr=0.001)
        loss = CategoricalCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.compiled = True
        print(self.model.summary())

    def predict(self, images):
        captions = []
        for image in images:
            initial = np.zeros((MAX_SEQ_LEN, VOCAB_SIZE))
            initial[-1, :] = 0
            initial[-1, 0] = 1

            for i in range(MAX_SEQ_LEN - 1, 0, -1):
                prediction = self.model.predict([np.array([image]), np.array([initial])])[0]
                initial = np.roll(initial, -1, axis=0)
                initial[-1, :] = prediction
                if np.argmax(prediction) == 1:
                    break
            captions.append(np.argmax(initial, axis=1))
        return captions

    def load(self, load_path):
        self.model = keras.models.load_model(load_path)

    def train(self, datagen, epochs, batch_size=10):
        if not self.compiled:
            self.compile()
        start_time = datetime.now()
        loss_history = []
        for epoch, (images, captions) in enumerate(datagen(batch_size=batch_size)):
            loss = 0
            accuracy = 0

            for idx, image in enumerate(images): 
                partial_captions, next_words = partial_generator(captions[idx])
                duped_images = np.stack((image,) * len(partial_captions), axis=0)
                partial_loss, partial_accuracy = self.model.train_on_batch([duped_images, partial_captions], next_words)
                loss += partial_loss
                accuracy += partial_accuracy

            loss /= batch_size
            accuracy /= batch_size
            elapsed_time = datetime.now() - start_time
            loss_history.append(loss)
            print('[Epoch {}/{}] [Loss: {}] [Acc%: {}] time: {}'.format(
                epoch,
                epochs,
                round(loss, 2),
                round(100 * accuracy),
                elapsed_time,
            ))

            if epoch >= epochs:
                break

        plt.plot(range(0, epochs + 1), loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_history.png')

        self.model.save('model_generator')

if __name__ == '__main__':
    model = SingleStageGeneratorV1()
    model.compile()
