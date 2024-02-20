#%%
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from liv_zones import crop

# %%

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
mito_paths = glob.glob(f'{path}/*/*/*/*/mitochondria_properties.csv', recursive=True)
nuclei_paths = glob.glob(f'{path}/*/*/*/*/nuclei_properties.csv', recursive=True)
ld_paths = glob.glob(f'{path}/*/*/*/*/lipid_droplet_properties.csv', recursive=True)
peroxi_paths = glob.glob(f'{path}/*/*/*/*/peroxisome_properties.csv', recursive=True)
portal_paths = glob.glob(f'{path}/*/*/*/*/portal_dist.npy', recursive=True)
central_paths = glob.glob(f'{path}/*/*/*/*/central_dist.npy', recursive=True)
#%%
organelles = ['nuclei', 'mito', 'ld', 'peroxi']
asinus = 0
multi_asinus_dataset = pd.DataFrame()
for asinus in range(len(peroxi_paths)):
    if asinus == 0:
        cell_count = 0
    else:
        cell_count = multi_asinus_dataset['cell_id'].max()
    # read csvs
    print(f'processing num {asinus}')
    nuclei_props = pd.read_csv(nuclei_paths[asinus])
    mito_props = pd.read_csv(mito_paths[asinus])
    ld_props = pd.read_csv(ld_paths[asinus])
    peroxi_props = pd.read_csv(peroxi_paths[asinus])

    portal_path = portal_paths[asinus]
    central_path = central_paths[asinus]

    scaled_x_cuts, crop_shape = vein_cutoffs(portal_path, central_path)

    # add org_type_labels
    nuclei_props['org_type'] = 1
    mito_props['org_type'] = 2
    ld_props['org_type'] = 3
    peroxi_props['org_type'] = 4

    # only keep keys that are shared between all organelles
    n_props = nuclei_props[['area', 'perimeter', 'centroid-0', 'centroid-1', 'aspect_ratio', 'boundry_dist', 'cell_id', 'org_type', 'ascini_position']]
    m_props = mito_props[['area', 'perimeter', 'centroid-0', 'centroid-1', 'aspect_ratio', 'boundry_dist', 'cell_id', 'org_type', 'ascini_position']]
    l_props = ld_props[['area', 'perimeter', 'centroid-0', 'centroid-1', 'aspect_ratio', 'boundry_dist', 'cell_id', 'org_type', 'ascini_position']]
    p_props = peroxi_props[['area', 'perimeter', 'centroid-0', 'centroid-1', 'aspect_ratio', 'boundry_dist', 'cell_id', 'org_type', 'ascini_position']]

    # normalize all individual properties 

    standardized_dataset = pd.concat([n_props, m_props, l_props, p_props])
    dataset = pd.concat([nuclei_props, mito_props, ld_props, peroxi_props])

    # overwrite specific props so that they are not included in the mean / std normalization
    standardized_dataset['centroid-0'] = (dataset['centroid-0']) / 22.187
    standardized_dataset['centroid-1'] = (dataset['centroid-1']) / 22.187
    standardized_dataset['org_type'] = dataset['org_type']
    standardized_dataset['cell_id'] = dataset['cell_id'] + cell_count
    standardized_dataset['asinus'] = asinus
    standardized_dataset['ascini_position'] = dataset['ascini_position']

    cut_1 = standardized_dataset['centroid-1'] > scaled_x_cuts[0] *crop_shape[1] / 22.187
    cut_2 = standardized_dataset['centroid-1'] < scaled_x_cuts[1] *crop_shape[1] / 22.187
    cutoff = pd.concat([cut_1, cut_2], axis=1)
    within_cutoff = cutoff.all(axis=1)
    standardized_dataset['cutoff'] = within_cutoff * 1

    multi_asinus_dataset = pd.concat([multi_asinus_dataset, standardized_dataset], axis=0)
#%%
multi_asinus_dataset.to_csv(f'datasets/organelle_dataset_02_14_24.csv')