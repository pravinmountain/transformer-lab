import torch 
import tiktoken
from config import load_config

class DataLoader:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.block_size = self.config.block_size
        self.batch_size = self.config.batch_size
        self.file_path = self.config.file_path
        self.encoding = self.config.encoding

    def load_data(self, file_path: str = None):
        file_path = file_path or self.file_path
        text = open(file_path, mode='r', encoding='utf-8').read()
        return text

    def encode(self, text: str):
        enc = tiktoken.get_encoding(self.encoding)
        return enc.encode(text)

    def decode(self, tokens):
        enc = tiktoken.get_encoding(self.encoding)
        return enc.decode(tokens)

    def train_val_split(self, data, split_ratio: float = 0.9, return_lengths: bool = False):
        n = int(split_ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]
        if return_lengths:
            return train_data, val_data, len(train_data), len(val_data)
        return train_data, val_data

    def get_batch(self, data):
        ix = torch.randint(0, len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

