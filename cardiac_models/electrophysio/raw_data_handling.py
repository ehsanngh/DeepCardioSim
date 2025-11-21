import meshio
import torch
import numpy as np

from torch_cluster import radius
from torch_geometric.data import Data

PLOC_NEIGHBOR_RADIUS = 0.75

def single_case_handling(file):
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
    
    ploc_bool = torch.tensor(case[:, 3:4], dtype=torch.float)
    pacing_indices = torch.where(ploc_bool.flatten() == 1)[0]

    for pacing_idx in pacing_indices:
        query_point = input_geom[pacing_idx].unsqueeze(0)
        row, col = radius(query_point, input_geom, r=PLOC_NEIGHBOR_RADIUS)
        ploc_bool[row] = 1.

    D_iso = torch.tensor(case[:, 4:5], dtype=torch.float)
    ef = torch.tensor(case[:, 5:8], dtype=torch.float)
    
    a = torch.cat((ploc_bool, D_iso, ef), axis=1)
    y = torch.tensor(case[:, -1:], dtype=torch.float).unsqueeze(2)
    
    if y.min() < 0.:
        return None

    try:
        label = file.split('case')[1].split('_nplocs')[0]
    except (IndexError, ValueError):
        label = None

    try:
        n_pacings = int(file.split('_nplocs')[1].split('.')[0])
    except (IndexError, ValueError):
        n_pacings = None

    data = Data(
        a=a,
        input_geom=input_geom,
        num_nodes=input_geom.size(0),
        y=y,
        label=label,
        n_pacings=n_pacings
        )
    if computed_labels is not None:
        data["computed_labels"] = torch.tensor(computed_labels, dtype=torch.float)
    return data
