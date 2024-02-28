# %%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.ops import MLP
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn

# %%
test_frac = 0.2
data = pd.read_csv("../../datasets/cell_dataset_02_21_24.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# %%
data.dropna(inplace=True)
# %%
labels = np.array(data["ascini_position"])
data = np.array(
    data.drop(
        [
            "Unnamed: 0",
            "ascini_position",
            "centroid-0",
            "centroid-1",
            "asinus",
            "cutoff",
        ],
        axis=1,
    )
)
test_rows = np.random.choice(
    np.arange(0, len(data)), int(test_frac * len(data)), replace=False
)

test_data = data[test_rows, :]
train_data = np.delete(data, test_rows, axis=0)

test_labels = np.array(labels)[test_rows]
train_labels = np.delete(labels, test_rows)

print(f" train: {train_data.shape}")
print(f" test: {test_data.shape}")


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

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# %%

device = torch.device("cpu")

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(train_data.shape[1], 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#     def forward(self, x):
#         out = self.mlp(x)
#         return out

model = MLP(
    in_channels=train_data.shape[1],
    hidden_channels=[32, 32, 32, 32, 1],
    activation_layer=torch.nn.ReLU,
)
# %%
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.MSELoss()
# %%
epochs = 5000

model.train()
losses = []
for epoch in tqdm(range(epochs)):
    for batch_num, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        x = data.to(device).float()
        y = labels.to(device).float()

        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        # losses.append(loss.item())

        optimizer.step()
    losses.append(loss.detach().numpy())

torch.save(model.state_dict(), "models/cell_mlp_022124")
np.save(f"models/cell_losses.npy", np.asarray(losses))
# %%
model.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(test_data)
    prediction_tensor = model(test_tensor.to(device).float())
prediction = prediction_tensor.detach().numpy()
# %%
plt.scatter(test_labels, prediction, alpha=0.8, s=5)
ax = plt.gca()
ax.set_aspect("equal")
plt.xlabel("True Acinus Position")
plt.ylabel("Predicted Acinus Position")
plt.figure()

plt.plot(np.arange(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# %%
model_path = "models/cell_mlp_022124"
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(test_data)
    prediction_tensor = model(test_tensor.to(device).float())
prediction = prediction_tensor.detach().numpy()
a = metrics.mean_squared_error(prediction[:, 0], test_labels)

# %%
