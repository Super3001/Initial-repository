import torch

check_ori = torch.load('model_lstm1_2.ckpt')

from lstm_1 import LSTM_1, MyDataset

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
print(device)
vocab_size = 30522
# vocab_size = 1000
hidden_dim = 80  # # mbed_dim
num_layers = 3  # [1, 10]
dropout = 0.1  # [0.1, 0.2]
num_classes = 28
model = LSTM_1(input_dim=vocab_size,
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           num_classes=num_classes,
                           dropout=dropout)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(type(check_ori))
model.load_state_dict(check_ori)
# optimizer.load_state_dict(check_ori['optimizer_state_dict'])
# epoch = check_ori['epoch']
# loss = check_ori['loss']

print(model)
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
ds = load_from_disk('./dataset')
train_dataset = MyDataset(ds['train'])
batch_size = 1
examine_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print('begin')
thresh = 10000
corr = 0
for i, (inputs, labels) in enumerate(examine_dataloader):
    if i % 1000 == 0:
        print(i)
    if i == thresh:
        break
    with torch.no_grad():
        outputs = model(inputs)
        if torch.argmax(outputs) == torch.argmax(labels):
            corr += 1
print(corr)
print('accuracy:', corr / thresh)


