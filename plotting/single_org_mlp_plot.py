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

#%%
model_path = 'models/peroxi_mlp_013124'
org_of_interest = 4
device='cpu'
test_frac = 0.05
dataset = pd.read_csv('../datasets/organelle_dataset_1_09_24.csv')
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)
# %%
single_org_data = dataset[dataset['org_type'] == org_of_interest]
labels = np.array(single_org_data['ascini_position'])
data = np.array(single_org_data.drop(['Unnamed: 0', 'ascini_position', 'centroid-0', 'centroid-1', 'cell_id', 'asinus', 'cutoff', 'org_type'], axis=1))
test_rows = np.random.choice(np.arange(0, len(data)), int(test_frac * len(data)), replace=False)

test_data = data[test_rows, :]
train_data = np.delete(data, test_rows, axis=0)

test_labels = np.array(labels)[test_rows]
train_labels = np.delete(labels, test_rows)

model = MLP(
    in_channels=4,
    hidden_channels=[32, 32, 32, 32, 1],
    activation_layer=torch.nn.ReLU
)
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(test_data)
    prediction_tensor = model(test_tensor.to(device).float())
prediction = prediction_tensor.detach().numpy()
# %%
plt.scatter(
    test_labels,
    prediction,
    alpha=0.1,
    s=5)
ax = plt.gca()
ax.set_aspect('equal')
plt.xlabel('True Acinus Position')
plt.ylabel('Predicted Acinus Position')
plt.ylim([-1,1])
plt.figure()
# %%
