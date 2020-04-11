import csv
import requests
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from socket import gaierror

num_to_download = 250
max_workers = 32

class Downloader():

    def __init__(self):
        self.session = FuturesSession(max_workers=max_workers)
        self.requests = []

    def get_urls(self):
        count = 0
        with open('train.tsv') as file:
            reader = csv.reader(file, delimiter='\t')
            for caption, url in reader:
                r = self.session.get(url)
                self.requests.append(r)
                count += 1
                if count >= num_to_download:
                    return

    def should_download_image(self, response):
        if response.status_code != 200:
            return False
        if 'content-type' not in response.headers:
            return False
        if response.headers['content-type'][:6] != 'image/':
            return False
        return True

    # todo - should i thread this
    def download_image(self, response):
        file_type = response.headers['content-type'][6:].split(';')[0]
        with open('data/raw/test.{}'.format(file_type), 'wb') as f:
            f.write(response.content)

    def collect_responses(self):
        for request in self.requests:
            try:
                response = request.result()
            except:
                continue
            if self.should_download_image(response):
                self.download_image(response)

downloader = Downloader()
downloader.get_urls()
downloader.collect_responses()