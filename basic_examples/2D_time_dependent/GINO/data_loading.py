import torch
import numpy as np
import glob
import os

from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer

from data_processing import CustomDataProcessorGINO
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset


class MeshDataset(Dataset):
    def __init__(self, data_list):
        super(MeshDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


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
        self.num_nodes = self.a.size(0)
        if key == 'edge_index':
            return torch.tensor([
                [self.latent_queries.size(1) * self.latent_queries.size(2)],
                [self.a.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
    

def load_dataset_gino(
        folder_path='./data',
        train_batch_sizes=[1],
        test_batch_sizes=[1],
        query_res=[32, 32],
        T=5,
        use_distributed=False):

    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

    meshes = []
    for file in npy_files:
        case = np.load(file)
        num_nodes = case.shape[0]
        num_timesteps = case[:, 4:].shape[1]
        
        input_geom = torch.tensor(case[:, 0:2], dtype=torch.float)

        spatial_features = torch.tensor(
            case[:, 2:4], dtype=torch.float).unsqueeze(1).expand(
                -1, num_timesteps, -1)
        
        drchlt_bl = spatial_features[:, :, 0:1]
        neumnn_bl = spatial_features[:, :, 1:2]

        y = torch.tensor(case[:, 4:], dtype=torch.float).unsqueeze(2)
        
        timesteps = torch.linspace(0, T, num_timesteps)
        
        src_term = (1 + torch.sin(timesteps / (0.5 * T) * torch.pi)
                    ) * 4 * (input_geom[:, 0:1] + 2 * input_geom[:, 1:2]) * 10

        a = torch.cat((drchlt_bl, neumnn_bl, src_term.unsqueeze(2)), axis=2)
        
        latent_queries = create_grid(
            input_geom.min(axis=0)[0],
            input_geom.max(axis=0)[0],
            query_res).unsqueeze(0)

        dists_1 = torch.cdist(latent_queries.reshape(-1, 2), input_geom)
        in_nbr = torch.where(dists_1 <= 0.25, 1., 0.).nonzero().T
        
        data = BipartiteData(
            a=a,
            input_geom=input_geom,
            latent_queries=latent_queries,
            edge_index=in_nbr,
            y=y,
            label=file[-8:-4]
            )
        
        meshes.append(data)
    
    a = torch.cat([data['a'] for data in meshes], dim=0)
    input_geom =  torch.cat([data['input_geom'] for data in meshes], dim=0)
    latent_queries =  torch.cat([data['latent_queries'] for data in meshes], dim=0)
    y = torch.cat([data['y'] for data in meshes], dim=0)
    
    a_encoder = DataScaler(dim=[0, 1])
    a_encoder.fit(a[:, :, -1:])

    input_geom_encoder = DataScaler(dim=[0])
    input_geom_encoder.fit(input_geom)

    latent_query_encoder = DataScaler(dim=[0, 1, 2])
    latent_query_encoder.fit(latent_queries)

    output_encoder = DataScaler(dim=[0, 1])
    output_encoder.fit(y)

    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler

        def distributed_dataloader(dataset: Dataset, batch_size: int):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,  # `DistributedSampler` handles shuffling
                sampler=DistributedSampler(dataset)
            )

        train_loader = distributed_dataloader(
            MeshDataset(meshes[:750]), batch_size=train_batch_sizes[0])
        
        test_loader = distributed_dataloader(
            MeshDataset(meshes[750:]), batch_size=test_batch_sizes[0])
    
    else:
        train_loader = DataLoader(
            MeshDataset(meshes[:750]), batch_size=train_batch_sizes[0], shuffle=True)
        
        test_loader = DataLoader(
            MeshDataset(meshes[750:]), batch_size=test_batch_sizes[0], shuffle=False)
    
    train_loaders = {0: train_loader}
    
    test_loaders = {0: test_loader}
    

    data_processor = CustomDataProcessorGINO(
        a_normalizer=a_encoder,
        input_geom_normalizer=input_geom_encoder,
        query_normalizer=latent_query_encoder,
        out_normalizer=output_encoder

    )
    return train_loaders, test_loaders, data_processor
