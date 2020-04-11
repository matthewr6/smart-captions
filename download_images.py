import csv
import requests
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from socket import gaierror

num_to_download = 2500
max_workers = 32
source_file = 'sources/train.tsv'
output_dir = 'data/raw/{}.{}'
fname_hash_len = 10

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
                    self.requests.append(r)
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

    def build_fname(self, url):
        return abs(hash(url)) % (10 ** fname_hash_len)

    # todo - should i thread this
    def download_image(self, response):
        ftype = response.headers['content-type'][6:].split(';')[0]
        fname = self.build_fname(response.url)
        with open(output_dir.format(fname, ftype), 'wb') as f:
            f.write(response.content)

    def collect_responses(self):
        # with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for request in self.requests:
            try:
                response = request.result()
            except:
                continue
            if self.valid_response(response):
                self.download_image(response)
                # future = executor.submit(self.download_image, response)
                # self.writers.append(future)

    # def wait_on_downloads(self):
    #     for writer in self.writers:
    #         writer.result()


downloader = Downloader()
downloader.get_urls()
downloader.collect_responses()
# downloader.wait_on_downloads()