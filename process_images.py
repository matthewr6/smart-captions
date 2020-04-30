import numpy as np
import glob
import os
import csv
from PIL import Image

raw_path = 'data/raw/*'
size = (250, 250)
processed_path = 'data/processed/{}'
raw_captions = 'data/captions_raw.csv'
index_file = 'data/captions.csv'

class ImageProcessor():

    def __init__(self):
        self.failed = []

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
            self.failed.append(idx)

    def process_all(self):
        for fname in glob.glob(raw_path):
            img_id = os.path.basename(fname).split('.')[0]
            if not os.path.isfile('data/processed/{}.npy'.format(img_id)):
                self.process_single_image(fname)

    def remove_failed_from_index(self):
        with open(raw_captions, 'r') as csv_in, open(index_file, 'w') as csv_out:
            writer = csv.writer(csv_out)
            for row in csv.reader(csv_in):
                if row[0] not in self.failed:
                    writer.writerow(row)


processor = ImageProcessor()
processor.process_all()
processor.remove_failed_from_index()
