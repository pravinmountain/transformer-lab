# I am using some of greatest scientific books for training and debugging.

import os
import requests

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

books = {
    # The periodic table
    'the-periodic-table'    : 'https://gist.github.com/pravinmountain/03072609dedb591eb3e588edcf78c2be/raw/the-periodic-table.txt',
    # Cosmos    
    'cosmos'                : 'https://gist.github.com/pravinmountain/e415905fe5372588d7f6844e0add99c6/raw/cosmos-carl-sagan.txt',
    # Origin of Species    
    'origin-of-species'     : 'https://gist.github.com/pravinmountain/aaa667dad90e9c17de0136ad2a5f6ce3/raw/origin-of-species.txt/origin_of_species_darwin.txt',
    # The Dragon of Eden     
    'the-dragon-of-eden'    : 'https://gist.github.com/pravinmountain/d49e413c442f4df6870a33800a6f0539/raw/the-dragons-of-eden-carl-sagan.txt',
}

if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
    for filename, url in books.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(f'data/{filename}.txt', 'wb') as f:
            f.write(response.content)

# The above cell downloaded 4 new text files in data folder.

text = open(file='data/cosmos.txt', mode='r', encoding='utf-8').read()

# number of unique characters in text, this is our vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# character to integer mapping and vice-versa
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for ch, i in stoi.items()}

# encoding string into list of integers
encode = lambda s: [stoi[ch] for ch in s]
# decoding list of integers into a string
decode = lambda l: "".join(itos[i] for i in l)

tokens = encode(text)

data = torch.tensor(tokens)

# split data into train and val split
# 90% train, 10% val
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

seq_len = 8
batch = 4
d_model = 32

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - seq_len, size=(batch,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, xb, yb):
        B, T = xb.shape     # 4, 8
        tok_embed = self.token_embeddings(xb) # tok_embed(4, 8, 32)
        pos_embed = self.positional_embeddings(torch.arange(T, device=xb.device)) # pos_embed(T, 32)
        x = tok_embed + pos_embed      # (B, T, 32)
        logits = self.lm_head(x)       # (B, T, vocab_size)
        logits = logits.view(B*T, vocab_size)   # (B*T, vocab_size)
        yb = yb.view(B*T)               # (4*8) --> (32,)
        loss = F.cross_entropy(logits, yb) # x(B*T, vocab_size), yb(B*T)
        return logits, loss 



xb, yb = get_batch(split='train')
model = Model()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)

for i in range(10):
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"loss: {loss.item()}")