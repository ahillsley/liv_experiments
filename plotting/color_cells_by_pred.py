# %%
import numpy as np
import torch
from torchvision.ops import MLP
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


path = "/groups/feliciano/felicianolab/For_Alex_and_Mark/After_median_filter"
cell_paths = glob.glob(f"{path}/*/*/*/*/cell_mask.npy", recursive=True)
del cell_paths[7]
del cell_paths[26]

predictions = pd.read_csv(
    "../liv_experiments/simple_models/predictions/cell_predictions_022324.csv"
)
data = pd.read_csv("../datasets/cell_dataset_02_21_24.csv")


# %%
def color_by_pred(num, predictions, cell_paths, true_pos=False):
    cell_mask = np.load(cell_paths[num])
    single_acinus = predictions[predictions["asinus"] == num]
    single_data = data[data["asinus"] == num]
    bad_cell_ids = np.nonzero(single_data.isna())[0]
    all_cells_in_mask = np.unique(cell_mask)[1:]
    pred_cells_in_mask = np.delete(all_cells_in_mask, bad_cell_ids)
    new_img = np.zeros((cell_mask.shape[0], cell_mask.shape[1]))

    if true_pos:
        pred_positions = np.array(single_acinus["ascini_position"])
    else:
        pred_positions = np.array(single_acinus["predicted_pos"])
    for i, j in enumerate(pred_cells_in_mask):
        # i is cell id in single_acinus
        # j is cell index in cell_mask
        pred = pred_positions[i]
        new_img[cell_mask == j] = pred
    return new_img


# %%

fig = plt.figure(figsize=(40, 12))
gs = fig.add_gridspec(5, 6)
cmap = plt.get_cmap("viridis")
cmap.set_bad("white")
cell_paths = cell_paths[:2]
for num in range(len(cell_paths)):
    if num == 21:
        continue
    img = color_by_pred(num, predictions, cell_paths, true_pos=True)
    img = np.ma.masked_equal(img, 0)
    ax = fig.add_subplot(gs[num])
    plt.imshow(img, cmap=cmap)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(f'true_position_plots/acini_{num}.png', dpi=300)
# %%

cmap = plt.get_cmap("viridis")
cmap.set_bad("white")
for num in range(len(cell_paths)):
    fig = plt.figure(figsize=(10, 3))
    if num == 21:
        continue
    img = color_by_pred(num, predictions, cell_paths, true_pos=False)
    img = np.ma.masked_equal(img, 0)
    ax = fig.add_subplot()
    plt.imshow(img, cmap=cmap)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"pred_position_plots/acini_{num}.png", dpi=300)
# %%
