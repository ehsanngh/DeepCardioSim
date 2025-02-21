import torch
import numpy as np
import glob
import os

from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer

from data_processing import CustomDataProcessorGINO
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from deepcardio.meshdata import MeshDataset, BipartiteData

from timeit import default_timer

DataScaler = UnitGaussianNormalizer

def create_grid(min_b, max_b, query_res):
    tx = torch.linspace(min_b[0], max_b[0], query_res[0])
    ty = torch.linspace(min_b[1], max_b[1], query_res[1])
    tz = torch.linspace(min_b[2], max_b[2], query_res[2])
    grid = torch.stack(
        torch.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            )
    return grid
    
def load_data_from_npy(
        folder_paths=['./data'],
        query_res=[32, 32, 32]):

    npy_files = []
    for folder_path in folder_paths:
        npy_files.extend(glob.glob(os.path.join(folder_path, '*.npy')))

    meshes = []
    for i, file in enumerate(npy_files):
        starttime = default_timer()
        case = np.load(file, allow_pickle=True)
        rng = np.random.default_rng()
        rng.shuffle(case, axis=0)
        input_geom = torch.tensor(case[:, :3], dtype=torch.float)

        num_nodes = input_geom.shape[0]
        num_timesteps = 1
        
        ploc_bool = case[:, 3:4]
        D_iso = case[:, 4:5]
        ef = case[:, 5:8]
        
        spatial_features = np.concatenate((ploc_bool, D_iso, ef), axis=1)
        a = torch.tensor(
            spatial_features, dtype=torch.float).unsqueeze(1).expand(
            num_nodes, num_timesteps, -1)

        y = torch.tensor(case[:, -1:], dtype=torch.float).unsqueeze(2)
        
        if y.min() < 0.:
            print(f'{i} {file} negative activation time.')
            continue
        
        latent_queries = create_grid(
            input_geom.min(axis=0)[0],
            input_geom.max(axis=0)[0],
            query_res).unsqueeze(0)
        
    
        dists_1 = torch.cdist(
            latent_queries.reshape(-1, latent_queries.size(-1)),
            input_geom)
        in_nbr = torch.where(dists_1 <= 0.25, 1., 0.).nonzero().T
        
        data = BipartiteData(
            a=a,
            input_geom=input_geom,
            latent_queries=latent_queries,
            edge_index=in_nbr,
            y=y,
            label=file.split('case')[1].split('.npy')[0]
            )
        meshes.append(data)
        print(f'{i} {file} loaded in {default_timer() - starttime} seconds.')

        if (i + 1) % 1000 == 0:
            torch.save(meshes, folder_paths[0][:-5] + '_processed/data.pt')
    
    torch.save(meshes, folder_paths[0][:-5] + '_processed/data.pt')
    return meshes

# load_data_from_npy(folder_paths=['./cardiac_examples/electrophysio/data/npy/'], query_res=[32, 32, 32])

def load_dataset_for_gino(
        folder_path='./data',
        train_batch_sizes=[1],
        test_batch_sizes=[1],
        use_distributed=False,
        dataset_format=BipartiteData):
    
    loadingdata_starttime = default_timer()
    dataset_format  # This line is necessary for BipartiteData to be loaded. Its coding style should be fixed.
    meshes = torch.load(
        folder_path, weights_only=False)
    print(len(meshes))
    a = torch.cat([data['a'] for data in meshes], dim=0)
    input_geom =  torch.cat([data['input_geom'] for data in meshes], dim=0)
    latent_queries =  torch.cat([data['latent_queries'] for data in meshes], dim=0)
    y = torch.cat([data['y'] for data in meshes], dim=0)

    a_encoder = DataScaler(dim=[0, 1])
    a_encoder.fit(a)

    input_geom_encoder = DataScaler(dim=[0])
    input_geom_encoder.fit(input_geom)

    latent_query_encoder = DataScaler(dim=[0, 1, 2])
    latent_query_encoder.fit(latent_queries)

    output_encoder = DataScaler(dim=[0, 1])
    output_encoder.fit(y)

    split_index = int(len(meshes) * 0.67)
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
            MeshDataset(meshes[:split_index]), batch_size=train_batch_sizes[0])
        
        test_loader = distributed_dataloader(
            MeshDataset(meshes[split_index:-400]), batch_size=test_batch_sizes[0])
        
        test_loader2 = distributed_dataloader(
            MeshDataset(meshes[-400:]), batch_size=test_batch_sizes[1])
    
    else:
        train_loader = DataLoader(
            MeshDataset(meshes[:split_index]),
            batch_size=train_batch_sizes[0], shuffle=True)
        
        test_loader = DataLoader(
            MeshDataset(meshes[split_index:-400]),
            batch_size=test_batch_sizes[0], shuffle=False)

        test_loader2 = DataLoader(
            MeshDataset(meshes[-400:]),
            batch_size=test_batch_sizes[1], shuffle=False)
    
    train_loaders = {0: train_loader}
    
    test_loaders = {0: test_loader, 1: test_loader2}
    

    data_processor = CustomDataProcessorGINO(
        a_normalizer=a_encoder,
        input_geom_normalizer=input_geom_encoder,
        query_normalizer=latent_query_encoder,
        out_normalizer=output_encoder

    )
    print(f'Data got loaded in {default_timer() - loadingdata_starttime} seconds.')
    return train_loaders, test_loaders, data_processor
