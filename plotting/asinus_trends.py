#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from liv_zones.utils import filter_cells


# %%
path = "../../../../../../feliciano/felicianolab/For_Alex_and_Mark/Male/CNT/"
cell_paths = glob.glob(f'{path}/*/*/*/*/average_properties_per_cell.csv', recursive=True)

prop = 'mito_aspect_ratio'
cell_props = pd.read_csv(cell_paths[0])
for path in cell_paths:
    temp_props = pd.read_csv(path)
    temp_props = (temp_props - temp_props.quantile(0.05)) / (temp_props.quantile(0.95) - temp_props.quantile(0.05))
    #trend = temp_props[prop].rolling(10, min_periods=1).mean()

cell_props['bin'] = np.floor(cell_props['ascini_position'] * 100) / 100
binned_props_raw = cell_props.groupby('bin').mean()
binned_props = binned_props_raw.rolling(10, min_periods=1).mean()
# %%
prop = 'type_2_peroxisome_avg_area'
path = cell_paths[1]
mean_df = pd.Series()
for path in cell_paths:
    if path[68:88] == 'Liv3/Lobule3/acinus1':
        continue
    if path[68:88] == 'Liv2/Lobule3/acinus2':
        continue
    temp_props = pd.read_csv(path)
    temp_props = filter_cells(temp_props).sort_values('ascini_position')
    #norm_props = (temp_props - temp_props.quantile(0.05)) / (temp_props.quantile(0.95) - temp_props.quantile(0.05))
    trend = temp_props.rolling(40, min_periods=5).mean()
    mean_df = pd.concat([mean_df, trend[prop]], axis=1)
    if path[71] == '1':
        color = 'tab:orange'
    if path[71] == '2':
        color = 'k'
    if path[71] == '3':
        color = 'tab:red'
    if path[71] == '5':
        color = 'tab:cyan'
    plt.plot(
        temp_props['ascini_position'],
        trend[prop],
        color=color
    )
    plt.title(prop)
    #plt.title(path[68:88])
    #plt.figure()
plt.plot(
        temp_props['ascini_position'],
        mean_df.mean(axis=1),
        color='k'
    )
# %%

