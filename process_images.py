import numpy as np
import glob
import os
from PIL import Image

raw_path = 'data/raw/*'
size = (400, 400)
processed_path = 'data/processed/{}'

class ImageProcessor():

    def __init__(self):
        pass

    def process_single_image(self, fname):
        idx = os.path.basename(fname).split('.')[0]
        try:
            img = Image.open(fname)
            img = img.resize(size)
            img = np.asarray(img)
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            np.save(processed_path.format(idx), img)
        except:
            print('{} failed'.format(fname))

    def process_all(self):
        for fname in glob.glob(raw_path):
            img_id = os.path.basename(fname).split('.')[0]
            if not os.path.isfile('data/processed/{}.npy'.format(img_id)):
                self.process_single_image(fname)

processor = ImageProcessor()
processor.process_all()
