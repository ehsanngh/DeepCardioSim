import torch
from timeit import default_timer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from deepcardio.meshdata import MeshDataset
DATAPROCESSOR_SAVEDIR = "./cardiac_models/electrophysio/"
from deepcardio.electrophysio import load_dataprocessor

def load_dataset(
        model_str="GINO",
        folder_path='./data',
        train_batch_sizes=[1],
        test_batch_sizes=[1],
        use_distributed=False,
        dataprocessor_dir=DATAPROCESSOR_SAVEDIR):

    dataprocessor_path = dataprocessor_dir + model_str + "/data_processor.pt"
    
    dloading_start = default_timer()

    meshes = []
    for p in range(1, 7):
        chunk = torch.load(
            f'{folder_path}/data_chunk_{p:03d}.pt', weights_only=False, mmap=True)
        meshes.extend(chunk)
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
                num_workers=2,
                sampler=DistributedSampler(dataset)
            )

        train_loader = distributed_dataloader(
            MeshDataset(meshes[:train_split_index]),
            batch_size=train_batch_sizes[0])
        
        test_loader = distributed_dataloader(
            MeshDataset(meshes[train_split_index:-test_split_index]),
            batch_size=test_batch_sizes[0])
        
        test_loader2 = distributed_dataloader(
            MeshDataset(meshes[-test_split_index:]),
            batch_size=test_batch_sizes[1])
    
    else:
        train_loader = DataLoader(
            MeshDataset(meshes[:train_split_index]),
            batch_size=train_batch_sizes[0],
            num_workers=2,
            shuffle=True)
        
        test_loader = DataLoader(
            MeshDataset(meshes[train_split_index:-test_split_index]),
            batch_size=test_batch_sizes[0],
            num_workers=2,
            shuffle=False)

        test_loader2 = DataLoader(
            MeshDataset(meshes[-test_split_index:]),
            batch_size=test_batch_sizes[1],
            num_workers=2,
            shuffle=False)

    train_loaders = {0: train_loader}
    test_loaders = {0: test_loader, 1: test_loader2}
    
    data_processor = load_dataprocessor(
        model_str, meshes, use_distributed, dataprocessor_path)
    print(f'Data got loaded in {default_timer() - dloading_start} seconds.')
    return train_loaders, test_loaders, data_processor