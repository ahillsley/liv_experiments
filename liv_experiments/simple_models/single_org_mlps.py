# %%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.ops import MLP
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

test_frac = 0.2
dataset = pd.read_csv('../datasets/organelle_dataset_1_09_24.csv')
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)
# %%
org_of_interest = 4
single_org_data = dataset[dataset['org_type'] == org_of_interest]
labels = np.array(single_org_data['ascini_position'])
data = np.array(single_org_data.drop(['Unnamed: 0', 'ascini_position', 'centroid-0', 'centroid-1', 'cell_id', 'asinus', 'cutoff', 'org_type'], axis=1))
test_rows = np.random.choice(np.arange(0, len(data)), int(test_frac * len(data)), replace=False)

test_data = data[test_rows, :]
train_data = np.delete(data, test_rows, axis=0)

test_labels = np.array(labels)[test_rows]
train_labels = np.delete(labels, test_rows)

print(f' train: {train_data.shape}')
print(f' test: {test_data.shape}')
# %%
class TrainDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.labels[ind]
        return x, np.expand_dims(y, axis=0)

train_set = TrainDataset(train_data, train_labels)

batch_size = 256*10
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# %%

model = MLP(
    in_channels=4,
    hidden_channels=[32, 32, 32, 32, 1],
    activation_layer=torch.nn.ReLU
)
# %%
device = torch.device("cuda")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.MSELoss()

epochs = 5000

model.train()
losses = []
start_time = time.time()
for epoch in tqdm(range(epochs)):
    
    for batch_num, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        x = data.to(device).float()
        y = labels.to(device).float()

        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        #losses.append(loss.item())

        optimizer.step()
    losses.append(loss.cpu().detach().numpy())
print('-'*50)
print(f'ran in {time.time()-start_time:.3f}')
print('-'*50)

torch.save(model.state_dict(),'models/peroxi_mlp_013124' )
