# saves wikitext dataset to a binary file for training.

import os
import numpy as np
import tiktoken
from tqdm import tqdm                     
from datasets import load_dataset          # huggingface datasets

enc = tiktoken.get_encoding("gpt2")
NUM_PROC = 8

def is_article_title(line: str) -> bool:
    # top-level headings look like " = Title = \n"; subsections are " = = Sub = = \n"
    s = line.strip()
    return len(s) > 4 and s.startswith("= ") and s.endswith(" =") \
        and not s.startswith("= =")

def process(batch):
    lines = batch["text"]
    all_ids = enc.encode_ordinary_batch(lines, num_threads=NUM_PROC)
    out = []
    for line, ids in zip(lines, all_ids):
        # EOT only at the start of a new article, not after every line
        out.append([enc.eot_token] + ids if is_article_title(line) else ids)
    return {"ids": out, "len": [len(x) for x in out]}

if __name__ == "__main__":
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    # drop empty rows
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0, num_proc=NUM_PROC)

    tokenized = dataset.map(
        process,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=NUM_PROC,
        desc="tokenizing the split",
    )

    dtype = np.uint16  # ok since enc.max_token_value == 50256 < 2**16

    for split, dset in tokenized.items():
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{split}.bin")
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        total_batches = min(1024, len(dset))   
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            if len(batch) == 0:
                continue
            arr_batch = np.concatenate(batch["ids"]).astype(dtype)
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        assert idx == arr_len, f"wrote {idx} tokens, expected {arr_len}"
        arr.flush()

