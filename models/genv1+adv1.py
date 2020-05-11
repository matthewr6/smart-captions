import numpy as np

from constants import seqs_to_captions
from data_generator import data_generator
from adversarial_v1 import AdversarialModelV1
from single_stage_generator_v1 import SingleStageGeneratorV1

# model = AdversarialModelV1(SingleStageGeneratorV1(), 'adv1+ssgv1')
model = SingleStageGeneratorV1()
model.train(data_generator, epochs=15, batch_size=25)
# model.compile()

gen = data_generator(batch_size=1)
images, captions = next(gen)

captions = np.argmax(captions, axis=2)
predictions = model.predict(images)
predicted_captions = seqs_to_captions(predictions)
print(seqs_to_captions(captions))
print(predictions)
print(predicted_captions) 