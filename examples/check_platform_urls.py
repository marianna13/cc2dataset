# End-to-end example of checking platform URLs from WAT files just to get training data for platform URLs classifier

from tqdm import tqdm
import pandas as pd
from yt_dlp import YoutubeDL
import multiprocessing.pool as pool
import warnings
from urllib.parse import urlparse
import numpy as np
from fastwarc import ArchiveIterator, WarcRecordType
import fsspec
import random
import simdjson
from urllib.parse import urljoin
warnings.filterwarnings("ignore")

class loggerOutputs:
    def error(msg):
        # print(msg)
        pass
    def warning(msg):
        pass
    def debug(msg):
        pass

def check_url(v_url):
  yt_args = {
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        "logger": loggerOutputs,
        "socket_timeout": 1
    }
  try:
      with YoutubeDL(yt_args) as ydl:
        info_dict = ydl.extract_info(v_url, download=False, process=False)
     
        return v_url, info_dict['formats'][0]['url'] # if it has a link then it's a video

  except Exception as err:
    return v_url, None

def flatten(l):
    return [item for sublist in l for item in sublist]

def check(urls):
    extracted_urls = []
    # print(pd.read_parquet('extracted_data.parquet')['info'].values)
    for url in tqdm(urls):
        extracted_urls.append(check_url(url))
    return extracted_urls

def make_link_absolute(url, base_url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    try:
        return urljoin(base_url, url)
    except ValueError:
        return url

def make_links_absolute(links, base_url):

    return [make_link_absolute(link["url"], base_url) for link in links if link.get('url')]


if __name__ == '__main__':
    num_wats = 10
    num_links_per_wat = 10_000
    wat_ids = random.sample(list(range(100)), num_wats)
    all_links = []
    for i in wat_ids:
        wat_links = []

        url = f'https://data.commoncrawl.org/crawl-data/CC-MAIN-2022-40/segments/1664030331677.90/wat/CC-MAIN-20220924151538-20220924181538-{i:05d}.warc.wat.gz'

        with fsspec.open(url, mode="rb", compression="gzip") as f:
                for record in ArchiveIterator(f, record_types=WarcRecordType.metadata, parse_http=False):
                    try:
                        record_data = simdjson.load(record.reader)  # type: ignore
                    except:  # pylint: disable=bare-except
                        continue
                    envelope = record_data["Envelope"]
                    payload = envelope["Payload-Metadata"]
                    if "HTTP-Response-Metadata" not in payload:
                        continue
                    http_resp = payload["HTTP-Response-Metadata"]
                    if "HTML-Metadata" not in http_resp:
                        continue
                    metadata = http_resp["HTML-Metadata"]
                    if "Links" not in metadata:
                        continue

                    links = metadata["Links"]
                    base_url = envelope["WARC-Header-Metadata"]["WARC-Target-URI"]
                    if "Head" in metadata and "Base" in metadata["Head"]:
                        try:
                            base_url = urljoin(base_url, metadata["Head"]["Base"])
                        except ValueError:
                            pass
                    links = make_links_absolute(links, base_url)
                    
                    wat_links.extend(links)

        all_links.extend(random.sample(wat_links, num_links_per_wat))

    pd.DataFrame(all_links, columns=['url']).to_parquet('all_links.parquet')

    
    n_proc = 64
    urls = np.array_split(all_links, n_proc)
    with pool.ThreadPool(n_proc) as p:
        processed_urls = flatten(p.map(check, urls))



    df = pd.DataFrame(processed_urls, columns=['url', 'extracted_url'])
    df['label'] = df['extracted_url'].apply(lambda x: 'positive' if x else 'negative')
    print(df.label.value_counts()) # just to see what we got
    df.to_parquet('checked_urls.parquet', engine='fastparquet')
    