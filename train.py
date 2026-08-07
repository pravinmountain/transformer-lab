import torch

from dataloader import DataLoader
from model import LanguageModel

data_loader = DataLoader('configs/deepseekv2.yaml')
data = data_loader.load_data()
train_data, val_data = data_loader.train_val_split(data)
x_train, y_train = data_loader.get_batch(train_data)
x_val, y_val = data_loader.get_batch(val_data)

model = LanguageModel(vocab_size=50257, block_size=data_loader.block_size, n_embd=32, n_layer=4, n_head=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    logits, loss = model(x_train, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_loss = model(x_val, y_val)

    print(f"Epoch: {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")