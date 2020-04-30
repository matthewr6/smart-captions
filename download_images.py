import csv
import requests
import glob
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from socket import gaierror

# so far, 0 - 17500

start_num = 0#12500
num_to_download = 17500
max_workers = 32
source_file = 'sources/train.tsv'
output_dir = 'data/raw/{}.{}'
word_subset_file = 'sources/word_subset.txt'
caption_output_file = 'data/captions_raw.csv'

class Downloader():

    def __init__(self):
        self.session = FuturesSession(max_workers=max_workers)
        self.requests = []
        self.writers = []
        self.captions = []
        with open(word_subset_file, 'r') as f:
            self.word_subset = set(f.read().splitlines())

    def image_is_downloaded(self, idx):
        for fname in glob.glob(output_dir.format(idx, '*')):
            if os.path.isfile(fname):
                return True
        return False

    def should_download_image(self, idx, url, caption):
        words = set(caption.split(' '))
        if not (words & self.word_subset):
            return False
        return True

    def get_urls(self):
        count = 0
        with open(source_file) as file:
            reader = csv.reader(file, delimiter='\t')
            for idx, (caption, url) in enumerate(reader):
                if self.should_download_image(idx, url, caption):
                    if count < start_num:
                        count += 1
                        continue
                    if self.image_is_downloaded(idx):
                        count += 1
                        self.captions.append((idx, caption))
                        if count >= num_to_download:
                            return
                        continue
                    r = self.session.get(url, timeout=2.5)
                    self.requests.append((r, idx, caption))
                    count += 1
                    if count % 100 == 0:
                        print(count)
                    if count >= num_to_download:
                        return

    def valid_response(self, response):
        if response.status_code != 200:
            return False
        if 'content-type' not in response.headers:
            return False
        if response.headers['content-type'][:6] != 'image/':
            return False
        return True

    def download_image(self, response, idx, caption):
        ftype = response.headers['content-type'][6:].split(';')[0]
        with open(output_dir.format(idx, ftype), 'wb') as f:
            f.write(response.content)
        self.captions.append((idx, caption))

    def collect_responses(self):
        count = 0 
        for request, original_idx, caption in self.requests:
            try:
                response = request.result()
            except Exception as e:
                print(e)
                continue
            if self.valid_response(response):
                count += 1
                self.download_image(response, original_idx, caption)
        print('Downloaded {} new images'.format(count))

    def save_captions(self):
        with open(caption_output_file, 'w') as out:
            csv_out = csv.writer(out)
            for row in self.captions:
                csv_out.writerow(row)


downloader = Downloader()
downloader.get_urls()
downloader.collect_responses()
downloader.save_captions()