#%%
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from liv_zones import crop

def vein_cutoffs(portal_path, central_path):
    portal_dist = np.load(portal_path)
    central_dist = np.load(central_path)

    portal_cm = np.mean(np.array(np.where(portal_dist ==0)), axis=1)
    central_cm = np.mean(np.array(np.where(central_dist ==0)), axis=1)
    
    x_cuts = np.array([central_cm[1], portal_cm[1]])
    scaled_x_cuts = x_cuts / portal_dist.shape[1]

    return scaled_x_cuts, portal_dist.shape

#%%
path = "/groups/feliciano/felicianolab/For_Alex_and_Mark/After_median_filter"
cell_paths = glob.glob(f'{path}/*/*/*/*/average_properties_per_cell.csv', recursive=True)
# create cell level dataset
cell_paths = glob.glob(f'{path}/*/*/*/*/average_properties_per_cell.csv', recursive=True)

#%%
kept_props = [
    'mito_density',
    'mito_avg_area',
    'mito_percent_total_area',
    'mito_distance_from_edge',
    'mito_aspect_ratio',
    'mito_solidity',
    'ld_density',
    'ld_avg_area',
    'ld_percent_total_area',
    'ld_distance_from_edge',
    'peroxisome_density',
    'peroxisome_avg_area',
    'peroxisome_percent_total_area',
    'peroxisome_distance_from_edge',
    'peroxisome_aspect_ratio',
    'peroxisome_solidity',
    'ascini_position',
    ]

asinus = 0
multi_asinus_cell_dataset = pd.DataFrame()
for asinus in range(len(peroxi_paths)):
    # read csvs
    print(f'processing num {asinus}')
    if 'Liv2' in cell_paths[asinus]:
        print(f'skipping path: {cell_paths[asinus]}')
        continue

    cell_props = pd.read_csv(cell_paths[asinus])

    portal_path = portal_paths[asinus]
    central_path = central_paths[asinus]

    scaled_x_cuts, crop_shape = vein_cutoffs(portal_path, central_path)

    c_props = cell_props[kept_props]
    print(c_props.mean())
    c_props = (c_props - c_props.mean()) / c_props.std()
    

    standardized_dataset = c_props
    standardized_dataset['centroid-0'] = cell_props['centroid-0'] / crop_shape[0]
    standardized_dataset['centroid-1'] = cell_props['centroid-1'] / crop_shape[1]
    standardized_dataset['asinus'] = asinus
    standardized_dataset['ascini_position'] = cell_props['ascini_position']

    cut_1 = standardized_dataset['centroid-1'] > scaled_x_cuts[0]
    cut_2 = standardized_dataset['centroid-1'] < scaled_x_cuts[1]
    cutoff = pd.concat([cut_1, cut_2], axis=1)
    within_cutoff = cutoff.all(axis=1)
    standardized_dataset['cutoff'] = within_cutoff * 1
    multi_asinus_cell_dataset = pd.concat([multi_asinus_cell_dataset, standardized_dataset], axis=0)
# %%
multi_asinus_cell_dataset.to_csv(f'cell_dataset_12_13_23.csv')
# %%