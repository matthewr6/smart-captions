from adversarial_v1 import AdversarialModelV1
from single_stage_generator_v1 import SingleStageGeneratorV1

model = AdversarialModelV1(SingleStageGeneratorV1(), 'adv1+ssgv1')
