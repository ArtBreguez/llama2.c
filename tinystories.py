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

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def create_query_response_pairs_from_csv(num_jsons=2):
# Load the CSV data using pandas
    csv_filename = "data/Customer-Support.csv"
    csv_data = pd.read_csv(csv_filename)

    # Calculate the number of examples per JSON
    examples_per_json = len(csv_data) // num_jsons

    for json_idx in range(num_jsons):
        query_response_pairs = []

        # Determine the start and end indices for the current JSON
        start_idx = json_idx * examples_per_json
        end_idx = (json_idx + 1) * examples_per_json if json_idx < num_jsons - 1 else len(csv_data)

        # Create JSON filename
        json_filename = f"data/query_response_dataset_{json_idx}.json"

        # Generate query-response pairs for the current JSON
        for _, row in csv_data.iloc[start_idx:end_idx].iterrows():
            query = row["query"]
            response = row["response"]
            formatted_example = {
                "story": f"Question: {response} answer: {query} <end_of_ans>",
                "instruction": {
                    "prompt": query,
                    "words": [],
                    "features": []
                },
                "summary": "",
                "source": "GPT-4"
            }
            query_response_pairs.append(formatted_example)

        # Write the JSON file
        with open(json_filename, "w") as f:
            json.dump(query_response_pairs, f, indent=4)

        print(f"Created query-response dataset {json_idx}: {json_filename}")


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
            text = example["story"]
            text = text.strip() # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

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
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
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