import numpy as np

from constants import seqs_to_captions
from data_generator import data_generator
from adversarial_v1 import AdversarialModelV1
from single_stage_generator_v1 import SingleStageGeneratorV1

# model = AdversarialModelV1(SingleStageGeneratorV1(), 'adv1+ssgv1')
model = SingleStageGeneratorV1()
model.train(data_generator, epochs=5000, batch_size=100)
# model.compile()

gen = data_generator(batch_size=5)
images, captions, next_words = next(gen)

single_predictions = model.single_predict(images, captions)
single_predictions = np.argmax(single_predictions, axis=1)
print(captions)
print(next_words)
print(single_predictions)

# predictions = model.predict(images)
# predicted_captions = seqs_to_captions(predictions)
# print(seqs_to_captions(captions))
# # print(predictions)
# print(predicted_captions) 