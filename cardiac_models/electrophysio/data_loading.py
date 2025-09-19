import torch
from timeit import default_timer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from deepcardio.meshdata import MeshDataset

DATAPROCESSOR_SAVEDIR = "./cardiac_models/electrophysio/"

def load_dataset(
        model="GINO",
        folder_path='./data',
        train_batch_sizes=[1],
        test_batch_sizes=[1],
        use_distributed=False,
        dataset_format=None,  # set explicitly when calling, e.g., BipartiteData or Data
        dataprocessor_dir=DATAPROCESSOR_SAVEDIR):

    dataprocessor_path = dataprocessor_dir + model + "/data_processor.pt"
    if model == "GINO":
        from GINO.gino_data_handling import load_dataprocessor
    elif model == "GNN":
        from GNN.gnn_data_handling import load_dataprocessor
    else:
        raise ValueError("Only 'GINO' or 'GNN' can be passed.")
    
    loadingdata_starttime = default_timer()
    if dataset_format is None:
        raise ValueError("You must specify a dataset_format.")
    dataset_format  # This triggers the necessary import/loading side effect

    meshes = torch.load(folder_path, weights_only=False)
    print(f'Total number of available samples: {len(meshes)}')

    train_split_index = int(len(meshes) * 0.50)
    test_split_index = int(len(meshes) * 0.25)
    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler

        def distributed_dataloader(dataset: Dataset, batch_size: int):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,  # DistributedSampler handles shuffling
                sampler=DistributedSampler(dataset)
            )

        train_loader = distributed_dataloader(
            MeshDataset(meshes[:train_split_index]), batch_size=train_batch_sizes[0])
        
        test_loader = distributed_dataloader(
            MeshDataset(meshes[train_split_index:-test_split_index]), batch_size=test_batch_sizes[0])
        
        test_loader2 = distributed_dataloader(
            MeshDataset(meshes[-test_split_index:]), batch_size=test_batch_sizes[1])
    
    else:
        train_loader = DataLoader(
            MeshDataset(meshes[:train_split_index]),
            batch_size=train_batch_sizes[0], shuffle=True)
        
        test_loader = DataLoader(
            MeshDataset(meshes[train_split_index:-test_split_index]),
            batch_size=test_batch_sizes[0], shuffle=False)

        test_loader2 = DataLoader(
            MeshDataset(meshes[-test_split_index:]),
            batch_size=test_batch_sizes[1], shuffle=False)
    
    train_loaders = {0: train_loader}
    test_loaders = {0: test_loader, 1: test_loader2}
    
    data_processor = load_dataprocessor(meshes, use_distributed, dataprocessor_path)
    print(f'Data got loaded in {default_timer() - loadingdata_starttime} seconds.')
    return train_loaders, test_loaders, data_processor