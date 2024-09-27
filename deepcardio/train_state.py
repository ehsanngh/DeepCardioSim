# Source: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/training/training_state.py

"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist


def load_training_state(save_dir: Union[str, Path], 
                        model: nn.Module,
                        load_best: bool = False,
                        optimizer: nn.Module=None,
                        scheduler: nn.Module=None,
                        regularizer: nn.Module=None,
                        map_location: dict=None) -> dict:
    
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    model : nn.Module
        model to save
    optimizer : nn.Module, optional
        optimizer object to save, by default None
    scheduler : nn.Module, optional
        scheduler object to save, by default None
    regularizer : nn.Module, optional
        regularizer object to save, by default None
    map_location : dict, optional
        mapping dictionary keyed `{device_from: device_to}`, by default None
        dictionary instructs torch to load a model from a checkpoint on rank `device_from`
        and send it to `device_to`

    Returns
    -------
    dict of training state
        keyed `{'model': model, etc}`
        
    """
    if not map_location:
        if dist.is_initialized():
            map_location = {"cuda:0" : f"cuda:{dist.get_rank}"}

    epoch = 1
    best_loss = torch.tensor(1E+12, dtype=torch.float)

    save_name = 'model'
    save_name_best_model = 'best_model'
    
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_path = save_dir.joinpath(f'{save_name}_snapshot_dict.pt')
    save_best_model_path = save_dir.joinpath(
        f'{save_name_best_model}_snapshot_dict.pt')

    if save_best_model_path.exists():
        best_snapshot = torch.load(
            save_best_model_path.as_posix(),
            map_location=map_location,
            weights_only=False)
        best_loss = best_snapshot["BEST_LOSS"]
        
        if load_best:
            model.load_state_dict(best_snapshot["MODEL_STATE"])
            print(f'[GPU{map_location[-1]}] Model loaded from snapshot at {save_best_model_path}')

    if save_path.exists():
        snapshot = torch.load(
            save_path.as_posix(),
            map_location=map_location,
            weights_only=False)
        epoch = snapshot["CURRENT_EPOCH"]

        if not load_best:
            model.load_state_dict(snapshot["MODEL_STATE"])
            print(f"[GPU{map_location[-1]}] Model loaded from snapshot at {save_path}")
        
        if optimizer is not None:
            optimizer.load_state_dict(snapshot["OPTIMIZER"])
        if scheduler is not None:
            scheduler.load_state_dict(snapshot["SCHEDULER"])
        if regularizer is not None:
            regularizer.load_state_dict(snapshot["REGULARIZER"])
    
    else:
        print((f"[GPU{map_location[-1]}] The file {save_path} does not exist. Model was not loaded"))
    
    return epoch, best_loss
    

def save_training_state(
        save_dir: Union[str, Path],
        epoch: int,
        model: nn.Module,
        save_best: bool = False,
        best_loss = None,
        optimizer: nn.Module = None,
        scheduler: nn.Module = None,
        regularizer: nn.Module = None) -> None:
    """save_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to save training state (model, optional optimizer, scheduler, regularizer)
    """
    snapshot = {
            "CURRENT_EPOCH": epoch,
            "MODEL_STATE": model.module.state_dict() \
                if hasattr(model, 'module') else model.state_dict(),
        }
    
    if save_best:
        save_name = 'best_model'
        if best_loss is None:
            raise ValueError("best_loss must be passed as input for saving best_model")
        snapshot["BEST_LOSS"] = best_loss
    else:
        save_name = 'model'
        if optimizer is not None:
            snapshot["OPTIMIZER"] = optimizer.state_dict()
        if scheduler is not None:
            snapshot["SCHEDULER"] = scheduler.state_dict()
        if regularizer is not None:
            snapshot["REGULARIZER"] = regularizer.state_dict()
    
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    save_path = save_dir.joinpath(f'{save_name}_snapshot_dict.pt').as_posix()
    torch.save(snapshot, save_path)
    gpu_id = str(next(model.parameters()).device)[-1]
    print(f"[GPU{gpu_id}] Successfully saved training state to {save_path}")