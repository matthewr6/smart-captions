import os
import csv
import glob

img_location = '../data/processed/*.*'
source_file = '../sources/train.tsv'
caption_output_file = '../data/captions.csv'

images = glob.glob(img_location)
images = [int(os.path.basename(i).split('.')[0]) for i in images]

total_images = max(images)

with open(source_file) as source, open(caption_output_file, 'w') as out:
    reader = csv.reader(source, delimiter='\t')
    writer = csv.writer(out)
    for idx, (caption, url) in enumerate(reader):
        if idx > total_images:
            break
        if idx in images:
            writer.writerow([idx, caption])
