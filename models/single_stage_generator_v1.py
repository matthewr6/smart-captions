import numpy as np
import tensorflow as tf

from datetime import datetime

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
)
from tensorflow.keras.optimizers import (
    RMSprop,
    # Adagrad,
    # Adadelta
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
    SimpleRNN
)

from tensorflow.keras.losses import (
    CategoricalCrossentropy
)

from data_generator import partial_generator

RecurrentLayer = GRU

MAX_SEQ_LEN = 30
# MAX_SEQ_LEN = 500
CHAR_SEQ_LEN = 500
NUM_CHARS = 29

from constants import VOCAB_SIZE

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

        pretrained_cnn = VGG16(weights='imagenet', include_top=False, input_tensor=img_input)
        flexible_layers = 1 # arbitrary
        for layer in pretrained_cnn.layers[:-flexible_layers]:
            layer.trainable = False

        # Dense from CNN
        img = Flatten()(pretrained_cnn.output)
        for nodes in [1024, 512, 256, 128]:
            img = Dense(nodes, activation='relu')(img)
            # img = Dropout(0.25)(img)
        img = Dense(64, activation='relu')(img)
        # img = Dropout(0.25)(img)

        previous_words = Input(shape=(MAX_SEQ_LEN, VOCAB_SIZE), name='gen_cur_words')
        self.inputs.append(previous_words)

        recurrent_layer = GRU(128)(previous_words)
        recurrent_layer = Flatten()(recurrent_layer)

        combined = Concatenate()([img, recurrent_layer])

        for nodes in [256, 512, VOCAB_SIZE]:
            combined = Dense(nodes, activation='relu')(combined)
            # combined = Dropout(0.25)(combined)

        combined = Dense(nodes, activation='softmax')(combined)
        self.outputs = [combined]

        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='single_stage_generator_v1')

    def compile(self):
        # rmsprop, adagrad, adadelta might be best options
        optimizer = RMSprop(lr=0.01)
        loss = CategoricalCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.compiled = True
        print(self.model.summary())

    def predict(self, images):
        initial = np.zeros((images.shape[0], MAX_SEQ_LEN, VOCAB_SIZE))
        for i in range(MAX_SEQ_LEN):
            predictions = self.model.predict([images, initial])
            initial[:, i, :] = predictions
        return np.argmax(initial, axis=2)

    def load(self, load_path):
        self.model = keras.models.load_model(load_path)

    def train(self, datagen, epochs, batch_size=10):
        if not self.compiled:
            self.compile()
        start_time = datetime.now()
        for epoch, (images, captions) in enumerate(datagen(batch_size=batch_size)):
            partial_captions, next_words = partial_generator(captions)
            loss = 0
            accuracy = 0
            for idx, image in enumerate(images):
                corresponding_partials = partial_captions[idx * MAX_SEQ_LEN:(idx + 1) * MAX_SEQ_LEN]
                corresponding_nexts = next_words[idx * MAX_SEQ_LEN:(idx + 1) * MAX_SEQ_LEN]
                duped_images = np.stack((image,)*MAX_SEQ_LEN, axis=0)
                # todo - better loss/acc computation probably
                partial_loss, partial_accuracy = self.model.train_on_batch([duped_images, corresponding_partials], corresponding_nexts)
                loss += partial_loss
                accuracy += partial_accuracy
            loss /= batch_size
            accuracy /= batch_size
            elapsed_time = datetime.now() - start_time
            print('[Epoch {}/{}] [Loss: {}] [Acc: {}] time: {}'.format(
                epoch,
                epochs,
                round(loss, 2),
                round(100 * accuracy),
                elapsed_time,
            ))

            if epoch >= epochs:
                break

        self.model.save('model_generator')

if __name__ == '__main__':
    model = SingleStageGeneratorV1()
    model.compile()
