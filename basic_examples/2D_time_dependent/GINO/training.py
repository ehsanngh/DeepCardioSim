# %%
from data_loading import load_dataset_gino
import torch
from deepcardio import Trainer
from deepcardio.losses import LpLoss, H1Loss
import os

use_distributed = "WORLD_SIZE" in os.environ

if use_distributed:
    print(f"Running in distributed mode on {int(os.environ["WORLD_SIZE"])} GPU(s).")
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")

else:
    print("Running in single GPU mode.")

dltrain, dltest, data_processor = load_dataset_gino(
    folder_path='examples/2D_time_dependent/data',
    train_batch_sizes=[15], test_batch_sizes=[15], query_res=[32, 32],
    use_distributed=use_distributed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_processor = data_processor.to(device)

l2loss = LpLoss(d=3, p=2)
h1loss = H1Loss(d=3)

train_loss = l2loss
eval_losses={'l2': l2loss}

from gino import GINO
model = GINO(
    in_channels=3,  # [dr_bl, nm_bl, src_trm]
    out_channels=1,
    gno_coord_dim=2,
    gno_coord_embed_dim=16,
    gno_radius=0.1,
    gno_transform_type='linear',
    fno_n_modes=[16, 16, 16],  # x_1, x_2, t
    fno_hidden_channels=64,
    fno_use_mlp=True,
    fno_norm='instance_norm',
    fno_ada_in_features=32,
    fno_factorization='tucker',
    fno_rank=0.4,
    fno_domain_padding=0.125,
    fno_mlp_expansion=1.0,
    fno_output_scaling_factor=1,
)
model = model.to(device)

# %%
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=50,
    min_lr=1e-6,
    mode="min")

# %%
trainer = Trainer(
    model=model,
    max_epochs=2000,
    device=device,
    data_processor=data_processor,
    amp_autocast=False,
    eval_interval=10,
    log_output=True,
    use_distributed=use_distributed,
    verbose=True,
    )

trainer.train(train_loader=dltrain[0],
              test_loaders=dltest,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses,
              save_every=10,
              save_best=True,
              best_loss_metric_key='0_l2',
              save_dir="examples/2D_time_dependent/GINO/ckpt",
              resume_from_dir="examples/2D_time_dependent/GINO/ckpt")

if use_distributed:
    dist.destroy_process_group()


