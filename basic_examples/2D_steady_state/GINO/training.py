# %%


from data_loading import load_dataset_gino
import torch
from deepcardio import Trainer
from deepcardio.losses import LpLoss, H1Loss

dltrain, dltest, data_processor = load_dataset_gino(
    folder_path='./data',
    train_batch_sizes=[10], test_batch_sizes=[10], query_res=[32, 32])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_processor = data_processor.to(device)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

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
    fno_n_modes=[16, 16],
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
    use_distributed=False,
    verbose=True,
    )

trainer.train(train_loader=dltrain[0],
              test_loaders=dltest,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses,
              save_every=100,
              save_dir="./GINO/ckpt",)
              # resume_from_dir="./GINO/ckpt")


