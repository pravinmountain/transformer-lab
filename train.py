from dataloader import DataLoader

data_loader = DataLoader('configs/deepseekv2.yaml')
data = data_loader.load_data()
train_data, val_data, train_len, val_len = data_loader.train_val_split(data)
x_train, y_train = data_loader.get_batch(train_data)
x_val, y_val = data_loader.get_batch(val_data)

