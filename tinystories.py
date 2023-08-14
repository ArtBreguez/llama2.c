"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "dataset"

def create_query_response_pairs_from_csv():
    query_response_pairs = []

    # Load the CSV data using pandas
    csv_filename = "dataset/Customer-Support.csv"
    csv_data = pd.read_csv(csv_filename)

    for _, row in csv_data.iterrows():
        query = row["query"]
        response = row["response"]
        query_response_pairs.append({"query": query, "response": response})

    new_dataset_filename = "dataset/query_response_dataset.json"
    with open(new_dataset_filename, "w") as f:
        json.dump(query_response_pairs, f, indent=4)

    print(f"Created query-response dataset: {new_dataset_filename}")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def pretokenize():
    enc = Tokenizer()

    def process_shard(shard):
        with open(shard, "r") as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data):
            query = example["query"]
            response = example["response"]
            
            query_tokens = enc.encode(query, bos=True, eos=False)  # encode the query, use BOS
            response_tokens = enc.encode(response, bos=False, eos=False)  # encode the response
            
            all_tokens.extend(query_tokens)
            all_tokens.extend(response_tokens)
            
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    shard_filenames = sorted(glob.glob(os.path.join("/home/alertrack/llama2.c/dataset", "*.json")))

    # process all the shards in a threadpool
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_shard, shard_filenames)

    print("Done.")

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        shard_filenames = sorted(glob.glob(os.path.join("/home/alertrack/llama2.c/dataset", "*.bin")))

        train_ratio = 0.8  # 80% for training, 20% for testing
        num_train_shards = int(len(shard_filenames) * train_ratio)

        if self.split == "train":
            shards_to_use = shard_filenames[:num_train_shards]
        else:
            shards_to_use = shard_filenames[num_train_shards:]

        while True:
            rng.shuffle(shards_to_use)
            for shard in shards_to_use:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y



class Task:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize", "create_query_response"])
    args = parser.parse_args()

    fun = {
        "download": download,
        "pretokenize": pretokenize,
        "create_query_response": create_query_response_pairs_from_csv,  # Adjusted function name
    }
    fun[args.stage]()