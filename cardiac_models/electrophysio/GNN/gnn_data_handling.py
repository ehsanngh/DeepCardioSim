import meshio
import torch
import numpy as np
import os

from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer

from deepcardio.electrophysio import EPDataProcessor

from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances

DataScaler = UnitGaussianNormalizer

def single_case_handling(file, nbr_radius=0.25):
    if file.endswith('.npy'):
        case = np.load(file, allow_pickle=True)
    elif file.endswith('.vtk'):
        mesh = meshio.read(file)
        case = np.concatenate(
            (mesh.points,
             np.array(mesh.point_data['ploc_bool'], dtype=np.float32).reshape((-1, 1)),
             np.array(mesh.point_data['D_iso'], dtype=np.float32).reshape((-1, 1)),
             np.array(mesh.point_data['ef'], dtype=np.float32),
             np.array(mesh.point_data['activation_time'], dtype=np.float32).reshape((-1, 1))), axis = 1)
    else:
        print(f"Unsupported file format: {file}")
        return None
    
    rng = np.random.default_rng()
    rng.shuffle(case, axis=0)
    input_geom = torch.tensor(case[:, :3], dtype=torch.float)
    
    ploc_bool = case[:, 3:4]
    D_iso = case[:, 4:5]
    ef = case[:, 5:8]
    
    spatial_features = np.concatenate((ploc_bool, D_iso, ef), axis=1)
    a = torch.tensor(spatial_features, dtype=torch.float)

    y = torch.tensor(case[:, -1:], dtype=torch.float)
    
    if y.min() < 0.:
        return None    

    pwd = pairwise_distances(input_geom)
    np.fill_diagonal(pwd, 1e+6)
    edge_index = torch.tensor(np.vstack(np.where(pwd <= nbr_radius)), dtype=torch.long)
    
    try:
        label = file.split('case')[1].split('_nplocs')[0]
    except (IndexError, ValueError):
        label = None  # or a default value like ""

    try:
        n_pacings = int(file.split('_nplocs')[1].split('.')[0])
    except (IndexError, ValueError):
        n_pacings = None

    data = Data(
        a=a,
        input_geom=input_geom,
        edge_index=edge_index,
        y=y.unsqueeze(1),
        num_nodes=input_geom.size(0),
        label=label,
        n_pacings=n_pacings
        )
    return data


def load_dataprocessor(meshes, use_distributed, dataprocessor_savepath, inference: bool = False):
    if os.path.exists(dataprocessor_savepath):
        print(f'data_processor loaded from {dataprocessor_savepath}.')
        return torch.load(dataprocessor_savepath, weights_only=False)
    
    if inference and not os.path.exists(dataprocessor_savepath):
        raise ValueError(f"""Data processor must exist for inference.
                         The path '{dataprocessor_savepath}' does not exist.""")
    
    rank = 0
    if use_distributed:
        import torch.distributed as dist
        rank = dist.get_rank()
        if rank != 0:
            print(f'cuda:{rank} waiting for data_processor to be created...')
            dist.barrier()  # Other processes wait until rank 0 is done
            return torch.load(dataprocessor_savepath, weights_only=False)
    
    print(f'Rank {rank} is creating the data_processor ...')
    a = torch.cat([data['a'] for data in meshes], dim=0)
    input_geom =  torch.cat([data['input_geom'] for data in meshes], dim=0)
    y = torch.cat([data['y'] for data in meshes], dim=0)
    
    a_encoder = DataScaler(dim=[0])
    a_encoder.fit(torch.concat((a, input_geom), axis=1))

    output_encoder = DataScaler(dim=[0, 1])
    output_encoder.fit(y)

    data_processor = EPDataProcessor(
        a_normalizer=a_encoder,
        input_geom_normalizer=None,
        out_normalizer=output_encoder

    )
    torch.save(data_processor, dataprocessor_savepath)
    if use_distributed:
        dist.barrier()
    return data_processor


