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
                        best_model: bool = False,
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
    if best_model:
        save_name = 'best_model'
    else:
        save_name = 'model'
        
    save_path = save_dir.joinpath(f'{save_name}_snapshot_dict.pt').as_posix()
    snapshot = torch.load(save_path, map_location=map_location)
    epoch = snapshot["CURRENT_EPOCH"]
    model.load_state_dict(snapshot["MODEL_STATE"])
    if optimizer is not None:
        optimizer.load_state_dict(snapshot["OPTIMIZER"])
    if scheduler is not None:
        scheduler.load_state_dict(snapshot["SCHEDULER"])
    if regularizer is not None:
        regularizer.load_state_dict(snapshot["REGULARIZER"])
    
    if best_model:
        best_loss = snapshot["BEST_LOSS"]
        print(f"Best model loaded from snapshot at {save_path}")
    else:
        best_loss = None
        print(f"Model loaded from snapshot at {save_path}")
    return epoch, best_loss
    

def save_training_state(
        save_dir: Union[str, Path],
        epoch: int,
        model: nn.Module,
        best_model: bool = False,
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
    if optimizer is not None:
        snapshot["OPTIMIZER"] = optimizer.state_dict()
    if scheduler is not None:
        snapshot["SCHEDULER"] = scheduler.state_dict()
    if regularizer is not None:
        snapshot["REGULARIZER"] = regularizer.state_dict()
    if best_model:
        save_name = 'best_model'
        if best_loss is None:
            raise ValueError("best_loss must be passed as input for saving best_model")
        snapshot["BEST_LOSS"] = best_loss
    else:
        save_name = 'model'
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    save_path = save_dir.joinpath(f'{save_name}_snapshot_dict.pt').as_posix()
    torch.save(snapshot, save_path)
    print(f"Successfully saved training state to {save_path}")