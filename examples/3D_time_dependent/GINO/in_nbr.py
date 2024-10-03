# In 2D examples, in_inbr information is calculated while loading the data.
# However, since this calculation is considerably time-consuming in 3D,
# this information is calculated beforehand using this script.

import torch
import numpy as np
import glob
import os

from timeit import default_timer

def create_grid(min_b, max_b, query_res):
    tx = torch.linspace(min_b[0], max_b[0], query_res[0])
    ty = torch.linspace(min_b[1], max_b[1], query_res[1])
    tz = torch.linspace(min_b[2], max_b[2], query_res[2])
    grid = torch.stack(
        torch.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            )
    return grid
    

def save_innbr(
        folder_path='./data',
        query_res=[32, 32, 32]):

    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    
    for i, file in enumerate(npy_files):
        case = np.load(file)        
        input_geom = torch.tensor(case[:, 0:3], dtype=torch.float)
        
        latent_queries = create_grid(
            input_geom.min(axis=0)[0],
            input_geom.max(axis=0)[0],
            query_res).unsqueeze(0)
        
        starttime = default_timer()
        dists_1 = torch.cdist(latent_queries.reshape(-1, latent_queries.size(-1)), input_geom)
        in_nbr = torch.where(dists_1 <= 0.25, 1., 0.).nonzero().T

        innbr_file = file.replace('/npy/', '/innbr/').replace('.npy', '.pt')
        torch.save(in_nbr, innbr_file)
        print(f'{i} in_nbr calculated in {default_timer() - starttime} seconds.')
    
    return None

save_innbr(folder_path='examples/3D_time_dependent/data/npy')