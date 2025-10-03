# %%
from data_loading import load_dataset
import torch
from deepcardio import Trainer
from deepcardio.losses import LpLoss
import os

from deepcardio.neuralop_core.training import AdamW
from deepcardio.neuralop_core.utils import count_model_params
from deepcardio.utils import determine_batch_size

from torch_geometric.seed import seed_everything
seed_everything(seed=12130875)

use_distributed = "WORLD_SIZE" in os.environ

if use_distributed:
    print(
        f"Running in distributed mode on {int(os.environ["WORLD_SIZE"])} GPU(s).")
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    os.environ["OMP_NUM_THREADS"] = str(2)

else:
    print("Running in single GPU/CPU mode.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

def main(model_str, batch_size):
    batch_size = batch_size \
        if device == 'cpu' else determine_batch_size(device, batch_size)
    
    if model_str == "GINO":
        from GINO.model import initialize_GINO_model
        model = initialize_GINO_model(n_fno_modes=16)
        folder_path='/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data_processed/data_GINO.pt'
        from deepcardio.meshdata import BipartiteData
        dataset_format=BipartiteData
        save_dir = "/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_models/electrophysio/GINO/ckpt"
        
    elif model_str == "GNN":
        from GNN.model import initialize_GNN_model
        model = initialize_GNN_model(size_hidden_layers=64)
        folder_path='/mnt/research/compbiolab/Ehsan/DeepCardioSim/cardiac_models/electrophysio/data_processed/data_GNN.pt'
        from torch_geometric.data import Data
        dataset_format=Data
        save_dir = "/mnt/home/naghavis/Documents/Research/DeepCardioSim/cardiac_models/electrophysio/GNN/ckpt"

    dltrain, dltest, data_processor = load_dataset(
        model=model_str,
        folder_path=folder_path,
        train_batch_sizes=[batch_size], test_batch_sizes=[batch_size, batch_size],
        use_distributed=use_distributed, dataset_format=dataset_format)


    dltest = {0: dltest[0]}
    data_processor = data_processor.to(device)

    l2loss = LpLoss(d=2, p=2, reductions='mean')

    train_loss = l2loss
    eval_losses={'l2': l2loss}

    model = model.to(device)
    
    print(f'number of trainable parameters: {count_model_params(model)}')

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=50,
        min_lr=1e-6,
        mode="min")
    
    trainer = Trainer(
        model=model,
        max_epochs=2000,
        device=device,
        data_processor=data_processor,
        amp_autocast=False,
        eval_interval=2,
        log_output=True,
        use_distributed=use_distributed,
        verbose=True,
        )

    trainer.train(
        train_loader=dltrain[0],
        test_loaders=dltest,
        optimizer=optimizer,
        scheduler=scheduler, 
        regularizer=False, 
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_every=2,
        save_best=True,
        best_loss_metric_key='0_l2',
        save_dir=save_dir,
        resume_from_dir=save_dir)

    if use_distributed:
        dist.destroy_process_group()

# %%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GINO", required=False)
    parser.add_argument('--batch_size', type=int, default=35, required=False)
    args = parser.parse_args()
    main(args.model, args.batch_size)