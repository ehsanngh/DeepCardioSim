from timeit import default_timer
import torch
import os
import glob
import argparse


def handle_npy_files(
        model,
        folder_paths=['./data'],
        outp_name='data.pt',
        query_res=[32, 32, 32],
        nbr_radius=0.25):

    if model == 'GINO':
        from GINO.gino_data_handling import single_case_handling
        kwargs = {'query_res': query_res}
    elif model == 'GNN':
        from GNN.gnn_data_handling import single_case_handling
        kwargs = {}
    else:
        raise ValueError("Only 'GINO' or 'GNN' can be passed.")
    
    outp_dir = folder_paths[0][:-5] + '_processed/' + outp_name
    npy_files = []
    for folder_path in folder_paths:
        npy_files.extend(glob.glob(os.path.join(folder_path, '*.npy')))

    meshes = []
    for i, file in enumerate(npy_files):
        starttime = default_timer()
        data = single_case_handling(file=file, nbr_radius=nbr_radius, **kwargs)
        if data is None:
            print(f'{i} {file} skipped due to negative activation time.')
            continue
        meshes.append(data)
        print(f'{i} {file} loaded in {default_timer() - starttime} seconds.')

        if (i + 1) % 1000 == 0:
            torch.save(meshes, outp_dir)
    
    torch.save(meshes, folder_paths[0][:-5] + outp_dir)
    return meshes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GINO", required=False)
    parser.add_argument('--outp_name', type=str, default="data.pt", required=False)
    args = parser.parse_args()
    
    handle_npy_files(
        args.model,
        folder_paths=['./cardiac_examples/electrophysio/data/npy/'],
        outp_name=args.outp_name,
        query_res=[32, 32, 32],
        nbr_radius=0.25)