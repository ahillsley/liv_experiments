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

# %%
# load model and format the dataset
model_path = "models/cell_mlp_022124"
device = "cpu"
dataset = pd.read_csv("../../datasets/cell_dataset_02_21_24.csv")
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)
dataset.dropna(inplace=True)
labels = np.array(dataset["ascini_position"])
data = np.array(
    dataset.drop(
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

# load trained model
model = MLP(
    in_channels=16,
    hidden_channels=[32, 32, 32, 32, 1],
    activation_layer=torch.nn.ReLU,
)
model.load_state_dict(torch.load(model_path))
# %%
# apply model to make predictions
model.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(data)
    prediction_tensor = model(test_tensor.to(device).float())
prediction = prediction_tensor.detach().numpy()
# %%
# append predictions to original dataset
dataset["predicted_pos"] = prediction
dataset.to_csv("predictions/cell_predictions_022324.csv")
