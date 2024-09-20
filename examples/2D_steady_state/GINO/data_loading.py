import torch
import numpy as np
import glob
import os

from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer

from data_processing import CustomDataProcessorGINO
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

DataScaler = UnitGaussianNormalizer


def create_grid(min_b, max_b, query_res):
    tx = torch.linspace(min_b[0], max_b[0], query_res[0])
    ty = torch.linspace(min_b[1], max_b[1], query_res[1])
    grid = torch.stack(
        torch.meshgrid(tx, ty, indexing="ij"), axis=-1
            )
    return grid


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([
                [self.latent_queries.size(1) * self.latent_queries.size(2)],
                [self.x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
    

def load_dataset_gino(
        folder_path='./myscripts/arbitrary_domain_GNO/data',
        train_batch_sizes=[1],
        test_batch_sizes=[1],
        query_res = [32, 32]
        ):

    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

    meshes = []
    for file in npy_files:
        case = np.load(file)
        input_geom = torch.tensor(case[:, :2], dtype=torch.float)
        latent_queries = create_grid(
            input_geom.min(axis=0)[0],
            input_geom.max(axis=0)[0],
            query_res).unsqueeze(0)
        drchlt_bl = torch.tensor(case[:, 2:3], dtype=torch.float)
        neumnn_bl = torch.tensor(case[:, 3:4], dtype=torch.float)
        src_term = 4 * input_geom[:, 0:1] + 2 * input_geom[:, 1:2]
        y = torch.tensor(case[:, -1:], dtype=torch.float)

        x = torch.cat((drchlt_bl, neumnn_bl, src_term), axis=1)
        
        dists_1 = torch.cdist(latent_queries.reshape(-1, 2), input_geom)
        in_nbr = torch.where(dists_1 <= 0.25, 1., 0.).nonzero().T
        data = BipartiteData(
            x=x,
            input_geom=input_geom,
            latent_queries=latent_queries,
            edge_index=in_nbr,
            y=y,
            label=file[-8:-4]
            )
        
        meshes.append(data)
    
    x = torch.cat([data['x'] for data in meshes], dim=0)
    input_geom =  torch.cat([data['input_geom'] for data in meshes], dim=0)
    latent_queries =  torch.cat([data['latent_queries'] for data in meshes], dim=0)
    y = torch.cat([data['y'] for data in meshes], dim=0)
    
    x_encoder = DataScaler(dim=[0])
    x_encoder.fit(x[:, -1:])

    input_geom_encoder = DataScaler(dim=[0])
    input_geom_encoder.fit(input_geom)

    latent_query_encoder = DataScaler(dim=[0, 1, 2])
    latent_query_encoder.fit(latent_queries)

    output_encoder = DataScaler(dim=[0])
    output_encoder.fit(y)

    train_loader = DataLoader(
        meshes[:750], batch_size=train_batch_sizes[0], shuffle=True)
    
    test_loader = DataLoader(
        meshes[750:], batch_size=test_batch_sizes[0], shuffle=False)
    
    train_loaders = {0: train_loader}
    
    test_loaders = {0: test_loader}
    

    data_processor = CustomDataProcessorGINO(
        x_normalizer=x_encoder,
        input_geom_normalizer=input_geom_encoder,
        query_normalizer=latent_query_encoder,
        out_normalizer=output_encoder

    )
    return train_loaders, test_loaders, data_processor
