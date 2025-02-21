import torch
import numpy as np
import glob
import os
from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer
from data_processing import CustomDataProcessorGraph
from sklearn.metrics import pairwise_distances


DataScaler = UnitGaussianNormalizer  # RangeNormalizer

import torch_geometric
from torch_geometric.data import Data

def calculate_edge_ftrs(pos, edge_index):

    if len(pos.shape) != 2:
        num_graphs, _, n1, n2 = pos.shape
        pos = pos.view(num_graphs, 2, -1).permute(0, 2, 1).reshape(-1, 2)
    diffs = pos[edge_index[1]] - pos[edge_index[0]]
    distances = torch.linalg.norm(diffs, dim=1, keepdim=True)
    return torch.concat((diffs / distances, distances), axis=1)

def load_dataset_graph(
        folder_path='./data',
        train_batch_sizes=[1], test_batch_sizes=[1], grad=False, seed=0):

    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

    graph_data =[]
    for file in npy_files:
        case = np.load(file)
        pos = torch.tensor(case[:, :2], dtype=torch.float)
        x = torch.tensor(case[:, 2:4], dtype=torch.float)
        y = torch.tensor(case[:, -1:], dtype=torch.float)
        pwd = pairwise_distances(case[:, :2])
        np.fill_diagonal(pwd, 1e+6)
        edge_index = torch.tensor(np.vstack(np.where(pwd <= 0.25)), dtype=torch.long)
        edge_ftrs = calculate_edge_ftrs(pos, edge_index)
        graph = Data(
             x=x, edge_index=edge_index, pos=pos,
             edge_attr=edge_ftrs, y = y, label=file[-8:-4])
        graph_data.append(graph)
    
    x = torch.cat([data.x for data in graph_data], dim=0) 
    pos = torch.cat([data.pos for data in graph_data], dim=0)
    y = torch.cat([data.y for data in graph_data], dim=0)
    edge_attr = torch.cat([data.edge_attr for data in graph_data], dim=0)
    
    input_encoder = DataScaler(dim=[0])
    input_encoder.fit(x)

    pos_encoder = DataScaler(dim=[0])
    pos_encoder.fit(pos)

    edge_attr_encoder = DataScaler(dim=[0])
    edge_attr_encoder.fit(edge_attr)

    output_encoder = DataScaler(dim=[0])
    output_encoder.fit(y)

    train_loader = torch_geometric.loader.DataLoader(
        graph_data[:750], batch_size=train_batch_sizes[0], shuffle=True)
    
    test_loader = torch_geometric.loader.DataLoader(
        graph_data[750:], batch_size=test_batch_sizes[0], shuffle=False)
    
    train_loaders = {0: train_loader}
    
    test_loaders = {0: test_loader}
    

    data_processor = CustomDataProcessorGraph(
        in_normalizer=input_encoder,
        edge_attr_normalizer=edge_attr_encoder,
        pos_normalizer=pos_encoder,
        out_normalizer=output_encoder

    )
    return train_loaders, test_loaders, data_processor

