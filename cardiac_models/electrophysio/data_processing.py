from deepcardio.neuralop_core.transforms import DataProcessor
from torch_geometric.data import Data
import torch
from torch_cluster import radius
from torch_scatter import scatter

import os
from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer
DataScaler = UnitGaussianNormalizer

GINO_NEIGHBOR_RADIUS = 0.5


def create_grids(pos, batch, query_res=(28, 28, 28)):
    """
    data.input_geom: (N, 3)
    data.batch:      (N,) graph IDs in [0, batch_size-1]
    Returns: (batch_size, nx, ny, nz, 3)
    """
    mins = scatter(pos, batch, dim=0, reduce='min') 
    maxs = scatter(pos, batch, dim=0, reduce='max')
    ranges = maxs - mins

    nx, ny, nz = query_res
    tx = torch.linspace(0, 1, nx, device=pos.device, dtype=pos.dtype)
    ty = torch.linspace(0, 1, ny, device=pos.device, dtype=pos.dtype)
    tz = torch.linspace(0, 1, nz, device=pos.device, dtype=pos.dtype)
    base = torch.stack(torch.meshgrid(tx, ty, tz, indexing="ij"), dim=-1)

    grids = mins[:, None, None, None, :] + ranges[:, None, None, None, :] * base[None, ...]

    return grids


class EPDataProcessor(DataProcessor):
    def __init__(
        self,
        model_str='GINO',
        a_normalizer=None,
        input_geom_normalizer=None,
        out_normalizer=None
    ):
        super().__init__()
        self.model_str = model_str
        self.a_normalizer = a_normalizer
        self.input_geom_normalizer = input_geom_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None
        self.query_res = (28, 28, 28)  # Used for GINO model. Should be consistent with model definition at ./GINO/model.py

    def to(self, device):
        if self.a_normalizer is not None:
            self.a_normalizer = self.a_normalizer.to(device)
        if self.input_geom_normalizer is not None:
            self.input_geom_normalizer = self.input_geom_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict_batch):
        data_dict_batch = data_dict_batch.to(self.device)
        a = data_dict_batch['a']
        input_geom = data_dict_batch['input_geom']
        num_nodes = input_geom.shape[0]

        if 'y' in data_dict_batch:
            y = data_dict_batch['y']
            if self.out_normalizer is not None:
                y = self.out_normalizer.transform(y)
            num_timesteps = y.shape[1]
        else:
            y = None
            num_timesteps = 1
        
        if 'batch' in data_dict_batch and data_dict_batch['batch'] is not None:
            batch = data_dict_batch['batch']
        else:  # single case
            batch = torch.zeros_like(input_geom[:, 0], dtype=torch.long) 
        if self.model_str == 'GINO':
            latent_queries = create_grids(input_geom, batch, query_res=self.query_res)
            batch_size, *grid_size, coord_dim = latent_queries.shape
            batch_y = batch
            batch_x = torch.arange(
                batch_size,
                device=latent_queries.device
            ).repeat_interleave(int(torch.tensor(grid_size).prod().item()))
            if self.training:
                max_num_neighbors = 32
            else:
                max_num_neighbors = 128
            row, col = radius(
                latent_queries.reshape(-1, latent_queries.size(-1)).contiguous(),
                input_geom.contiguous(),
                batch_x=batch_x,
                batch_y=batch_y,
                r=GINO_NEIGHBOR_RADIUS,
                max_num_neighbors=max_num_neighbors)
            edge_index = torch.stack([col, row], dim=0)
        else:  # GNN model
            edge_index = None  # GNN model computes edge_index in the forward pass
            a = torch.concat((a, input_geom), axis=1)


        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.transform(input_geom)
        
        if self.a_normalizer is not None:
            a = self.a_normalizer.transform(a)
        
        data_dict_batch_updated = Data(
            a=a.unsqueeze(1).expand(num_nodes, num_timesteps, -1) if self.model_str == 'GINO' else a,
            input_geom=input_geom,
            edge_index=edge_index.int() if edge_index is not None else None,
            y=y)
        
        return data_dict_batch.update(data_dict_batch_updated)

    def postprocess(self, output, data_dict_batch):
        data_dict_batch = data_dict_batch.to(self.device)
        a = data_dict_batch['a']
        input_geom = data_dict_batch['input_geom']

        if self.model_str == 'GINO':
            a = a.squeeze(1)

        if self.a_normalizer is not None:
            a = self.a_normalizer.inverse_transform(a)
        if self.input_geom_normalizer is not None:
            input_geom = self.input_geom_normalizer.inverse_transform(input_geom)

        if self.model_str == 'GNN':
            a = a[..., :-3]

        if 'y' in data_dict_batch:
            y = data_dict_batch['y']
        else:
            y = None
            
        if self.out_normalizer is not None and not self.training:
            output = self.out_normalizer.inverse_transform(output)
            if y is not None:
                y = self.out_normalizer.inverse_transform(y)
                
        data_dict_batch_updated = Data(
            a=a,
            input_geom=input_geom,
            y=y)
        
        return output, data_dict_batch.update(data_dict_batch_updated)

    def forward(self, **data_dict_batch):
        data_dict_batch = self.preprocess(data_dict_batch)
        output = self.model(**data_dict_batch)
        output = self.postprocess(output, data_dict_batch)
        return output, data_dict_batch


def load_dataprocessor(model_str, meshes, use_distributed, dataprocessor_savepath):
    if os.path.exists(dataprocessor_savepath):
        print(f'data_processor loaded from {dataprocessor_savepath}.')
        return torch.load(dataprocessor_savepath, weights_only=False)
    
    rank = 0
    if use_distributed:
        import torch.distributed as dist
        rank = dist.get_rank()
        if rank != 0:
            print(f'cuda:{rank} waiting for data_processor to be created...')
            dist.barrier()
            return torch.load(dataprocessor_savepath, weights_only=False)
    
    print(f'Rank {rank} is creating the data_processor ...')
    a = torch.cat([data['a'] for data in meshes], dim=0)
    input_geom =  torch.cat([data['input_geom'] for data in meshes], dim=0)
    y = torch.cat([data['y'] for data in meshes], dim=0)

    if model_str == 'GNN':
        a = torch.cat((a, input_geom), axis=1)
        input_geom_encoder = None
    else: # GINO
        input_geom_encoder = DataScaler(dim=[0])
        input_geom_encoder.fit(input_geom)

    a_encoder = DataScaler(dim=[0])
    a_encoder.fit(a)

    output_encoder = DataScaler(dim=[0, 1])
    output_encoder.fit(y)

    data_processor = EPDataProcessor(
        model_str=model_str,
        a_normalizer=a_encoder,
        input_geom_normalizer=input_geom_encoder,
        out_normalizer=output_encoder
    )

    torch.save(data_processor, dataprocessor_savepath)
    if use_distributed:
        dist.barrier()
    return data_processor