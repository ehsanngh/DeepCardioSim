from timeit import default_timer
import torch
import os
import glob
from pathlib import Path
from raw_data_handling import single_case_handling


def handle_npy_files(
        folder_paths=['/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/'],
        chunk_size=6000):
    CHUNK_SIZE = chunk_size
    
    parent_dir = Path(folder_paths[0]).parent.parent
    outp_dir = parent_dir / 'data_processed'
    outp_dir.mkdir(parents=True, exist_ok=True)

    npy_files = []
    for folder_path in folder_paths:
        npy_files.extend(glob.glob(os.path.join(folder_path, '*.npy')))

    meshes = []

    for i, file in enumerate(npy_files):
        starttime = default_timer()
        data = single_case_handling(file=file)
        if data is None:
            print(f'{i} {file} skipped due to negative activation time.')
            continue
        meshes.append(data)
        print(f'{i} {file} loaded in {default_timer() - starttime} seconds.')
        if (i + 1) % CHUNK_SIZE == 0:
            chunk_idx = (i + 1) // CHUNK_SIZE
            torch.save(meshes, outp_dir / f"data_chunk_{chunk_idx:03d}.pt")
            meshes = []
    
    if meshes:
        chunk_idx += 1
        torch.save(meshes, outp_dir / f"data_chunk_{chunk_idx:03d}.pt")
    
    return None


if __name__ == '__main__':
    
    handle_npy_files(
        folder_paths=['/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data/npy/'],
        chunk_size=6000)