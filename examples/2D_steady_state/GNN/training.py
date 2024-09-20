# %%
from data_loading import load_dataset_graph
import torch
from deepcardio import Trainer
from deepcardio.losses import LpLoss, H1Loss

dltrain, dltest, data_processor = load_dataset_graph(
    folder_path='./data', train_batch_sizes=[10], test_batch_sizes=[10])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_processor = data_processor.to(device)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = l2loss
eval_losses={'l2': l2loss}

from gnn_model import KernelNN
model = KernelNN(node_prj_dim=32, edge_prj_dim=24, num_layers=6, edge_attrs_dim=3, node_ftrs_dim=5, out_dim=1)
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
    eval_interval=100,
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
              save_dir="./GNN/ckpt"),
              # resume_from_dir="./myscripts/arbitrary_domain_GNO/model/ckpt")