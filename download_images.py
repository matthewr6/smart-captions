import csv
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from socket import gaierror

num_to_download = 2500
max_workers = 32
source_file = 'sources/train.tsv'
output_dir = 'data/raw/{}.{}'

class Downloader():

    def __init__(self):
        self.session = FuturesSession(max_workers=max_workers)
        self.requests = []
        self.writers = []

    def should_download_image(self, caption):
        words = caption.split(' ')
        return True

    def get_urls(self):
        count = 0
        with open(source_file) as file:
            reader = csv.reader(file, delimiter='\t')
            for caption, url in reader:
                if self.should_download_image(caption):
                    r = self.session.get(url, timeout=2.5)
                    self.requests.append((r, url, caption))
                    count += 1
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

    def build_fname(self, url, caption):
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def download_image(self, response, url, caption):
        ftype = response.headers['content-type'][6:].split(';')[0]
        fname = self.build_fname(url, caption)
        with open(output_dir.format(fname, ftype), 'wb') as f:
            f.write(response.content)

    def collect_responses(self):
        count = 0 
        for request, url, caption in self.requests:
            try:
                response = request.result()
            except:
                continue
            if self.valid_response(response):
                count += 1
                self.download_image(response, url, caption)
        print('Downloaded {} images'.format(count))


downloader = Downloader()
downloader.get_urls()
downloader.collect_responses()