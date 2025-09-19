import meshio
import torch
import numpy as np
import os

from deepcardio.neuralop_core.transforms import UnitGaussianNormalizer
from deepcardio.electrophysio import EPDataProcessor

from deepcardio.meshdata import BipartiteData

DataScaler = UnitGaussianNormalizer

def create_grid(min_b, max_b, query_res):
    tx = torch.linspace(min_b[0], max_b[0], query_res[0])
    ty = torch.linspace(min_b[1], max_b[1], query_res[1])
    tz = torch.linspace(min_b[2], max_b[2], query_res[2])
    grid = torch.stack(
        torch.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            )
    return grid


def single_case_handling(file, query_res=[32, 32, 32], nbr_radius=0.25):
    computed_labels = None
    if file.endswith('.npy'):
        case = np.load(file, allow_pickle=True)
    elif file.endswith('.vtk'):
        mesh = meshio.read(file)
        arrays_to_concat = [mesh.points]
        
        if 'ploc_bool' in mesh.point_data:
            arrays_to_concat.append(
                np.array(
                    mesh.point_data['ploc_bool'],
                    dtype=np.float32).reshape((-1, 1)))
        else:
            arrays_to_concat.append(
                np.zeros((mesh.points.shape[0], 1), dtype=np.float32))
            
        if 'D_iso' in mesh.point_data:
            arrays_to_concat.append(
                np.array(
                    mesh.point_data['D_iso'],
                    dtype=np.float32).reshape((-1, 1)))
        else:
            arrays_to_concat.append(
                np.zeros((mesh.points.shape[0], 1), dtype=np.float32))
            
        if 'ef' in mesh.point_data:
            arrays_to_concat.append(
                np.array(mesh.point_data['ef'], dtype=np.float32))
        else:
            arrays_to_concat.append(
                np.zeros((mesh.points.shape[0], 3), dtype=np.float32))
            
        if "activation_time" in mesh.point_data:
            arrays_to_concat.append(
                np.array(
                    mesh.point_data["activation_time"],
                    dtype=np.float32).reshape((-1, 1)))
        elif "t_act" in mesh.point_data:
            arrays_to_concat.append(
                np.array(
                    mesh.point_data["t_act"],
                    dtype=np.float32).reshape((-1, 1)))
        else:
            arrays_to_concat.append(
                np.zeros((mesh.points.shape[0], 1), dtype=np.float32))
            
        if "computed_labels" in mesh.point_data:
            computed_labels = np.array(
                    mesh.point_data["computed_labels"],
                    dtype=np.float32).reshape((-1, 1))
            
        case = np.concatenate(arrays_to_concat, axis=1)
    else:
        print(f"Unsupported file format: {file}")
        return None
    
    rng = np.random.default_rng()
    # Create permutation index for consistent shuffling
    perm_idx = rng.permutation(case.shape[0])
    case = case[perm_idx]
    if computed_labels is not None:
        computed_labels = computed_labels[perm_idx]
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
        return None
    
    latent_queries = create_grid(
        input_geom.min(axis=0)[0],
        input_geom.max(axis=0)[0],
        query_res).unsqueeze(0)
    
    dists_1 = torch.cdist(
        latent_queries.reshape(-1, latent_queries.size(-1)),
        input_geom)
    in_nbr = torch.where(dists_1 <= nbr_radius, 1., 0.).nonzero().T
    
    try:
        label = file.split('case')[1].split('_nplocs')[0]
    except (IndexError, ValueError):
        label = None

    try:
        n_pacings = int(file.split('_nplocs')[1].split('.')[0])
    except (IndexError, ValueError):
        n_pacings = None

    data = BipartiteData(
        a=a,
        input_geom=input_geom,
        latent_queries=latent_queries,
        edge_index=in_nbr,
        y=y,
        label=label,
        n_pacings=n_pacings
        )
    if computed_labels is not None:
        data["computed_labels"] = torch.tensor(computed_labels, dtype=torch.float)
    return data

def load_dataprocessor(meshes, use_distributed, dataprocessor_savepath):
    if os.path.exists(dataprocessor_savepath):
        print(f'data_processor loaded from {dataprocessor_savepath}.')
        return torch.load(dataprocessor_savepath, weights_only=False)
    
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

    data_processor = EPDataProcessor(
        a_normalizer=a_encoder,
        input_geom_normalizer=input_geom_encoder,
        query_normalizer=latent_query_encoder,
        out_normalizer=output_encoder
    )

    torch.save(data_processor, dataprocessor_savepath)
    if use_distributed:
        dist.barrier()
    return data_processor


