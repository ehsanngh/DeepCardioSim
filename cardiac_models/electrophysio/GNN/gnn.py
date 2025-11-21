# Source: https://github.com/Extrality/ICLR_NACA_Dataset_V0/blob/main/models/GUNet.py

import torch
import torch.nn as nn
import torch_geometric.nn as nng

import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity
from torch_geometric.nn import Linear


class MLP(torch.nn.Module):
    r"""A multi-layer perception (MLP) model.

    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels. :obj:`len(channel_list) - 1` denotes the number of layers
            of the MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(self, channel_list, dropout = 0.,
                 batch_norm = True, relu_first = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(self.channel_list[:-1], self.channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(self.channel_list[1:-1]):
            self.norms.append(BatchNorm1d(
                dim, 
                track_running_stats = True,
                allow_single_element = True) if batch_norm else Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = lin.forward(x)
        return x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


def DownSample(
    id,
    x,
    edge_index,
    pos_x,
    pool,
    pool_ratio,
    r,
    training,
    batches,
    ptrs,
    ploc_bool=None):
    y = x.clone()
    n_batch = len(ptrs[-1]) - 1
    n_nodes_per_batch = ptrs[-1][1:] - ptrs[-1][:-1]
    max_n = n_nodes_per_batch.max().item()

    if pool is not None:
        y, _, _, _, id_sampled, _ = pool(y, edge_index)
    else:
        k = (pool_ratio*n_nodes_per_batch).ceil().long()
        max_k = k.max().item()
        
        indices = torch.arange(
            max_n, device=x.device
            ).unsqueeze(0).expand(n_batch, max_n)  # creates a tensor of [[0, 1, ..., max_n] * n_batch]
        mask = indices < n_nodes_per_batch.unsqueeze(1)  # returns False for the extra nodes
        rand_vals = torch.rand(n_batch, n_nodes_per_batch.max(), device=x.device)

        if ploc_bool is not None:
            ploc_flat = ploc_bool.flatten()
            for b_idx in range(n_batch):
                batch_start = ptrs[-1][b_idx].item()
                batch_end = ptrs[-1][b_idx + 1].item()
                batch_ploc = ploc_flat[batch_start:batch_end]
                pacing_mask_local = batch_ploc > 0.5
                keep_target = int((pacing_mask_local.sum().item() + 4) // 5)  # At least 20% preserving pacing points
                if keep_target > 0:
                    pacing_indices = torch.nonzero(
                        pacing_mask_local, as_tuple=False).flatten()
                    perm = torch.randperm(
                        pacing_mask_local.sum().item(), device=x.device)
                    keep_local = pacing_indices[perm[:keep_target]]
                    # Set very small values to ensure they are selected first in sorting
                    rand_vals[b_idx, keep_local] = -1.0
        rand_vals[~mask] = 2.0  # keeps extra nodes at the far right even after sorting
        
        _, sorted_indices = torch.sort(rand_vals, dim=1)
        
        selected_indices = (sorted_indices + ptrs[-1][:-1].unsqueeze(1))[:, :max_k]
        
        mask = (indices < k.unsqueeze(1))[:, :max_k]
        id_sampled = selected_indices[mask]
        y = y[id_sampled]

    pos_x = pos_x[id_sampled]
    id.append(id_sampled)
    new_batch = batches[-1][id_sampled]
    new_ptr = torch.cat(
        [torch.tensor([0], device=x.device),
         torch.bincount(new_batch).cumsum(dim=0)])

    batches.append(new_batch)
    ptrs.append(new_ptr)

    if training:
        edge_index_sampled = nng.radius_graph(
            x = pos_x.detach(),
            r = r,
            loop = True,
            max_num_neighbors = 32,
            batch=new_batch)
    else:
        edge_index_sampled = nng.radius_graph(
            x = pos_x.detach(),
            r = r,
            loop = True,
            max_num_neighbors = 128,
            batch=new_batch)

    return y, edge_index_sampled

def UpSample(x, pos_x_up, pos_x_down, batch_x, batch_y):
    cluster = nng.nearest(
        x=pos_x_up, y=pos_x_down, batch_x=batch_x, batch_y=batch_y)
    x_up = x[cluster]

    return x_up

class GraphUNet(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            size_hidden_layers = 8,
            mlp_hidden_channels = 64,
            mlp_depth = 2,
            dim_enc = 8,
            depth: int = 5,
            layer = 'SAGE',
            pool = 'random',
            pool_ratio = [.75, .75, .666, .666],
            list_r = [0.5, .75, 1.25, 1.75, 2.5],
            bn_bool = False,
            ):
        super(GraphUNet, self).__init__()

        self.L = depth
        self.layer = layer
        self.pool_type = pool
        self.pool_ratio = pool_ratio
        self.r_first = list_r[0]
        self.list_r = list_r[1:]
        self.size_hidden_layers = size_hidden_layers
        self.size_hidden_layers_init = size_hidden_layers
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dim_enc = dim_enc
        self.bn_bool = bn_bool
        self.res = False
        self.head = 2
        self.activation = nn.ReLU()

        self.encoder = MLP(
            [in_channels] + mlp_depth * [mlp_hidden_channels] + [dim_enc],
            batch_norm = False)
        
        self.decoder = MLP(
            [dim_enc] + \
                  mlp_depth * [mlp_hidden_channels] + [out_channels],
            batch_norm = False)

        self.down_layers = nn.ModuleList()

        if self.pool_type != 'random':
            self.pool = nn.ModuleList()
        else:
            self.pool = None

        if self.layer == 'SAGE':
            self.down_layers.append(nng.SAGEConv(
                in_channels = self.dim_enc,
                out_channels = self.size_hidden_layers
            ))
            bn_in = self.size_hidden_layers

        elif self.layer == 'GAT':
            self.down_layers.append(nng.GATConv(
                in_channels = self.dim_enc,
                out_channels = self.size_hidden_layers,
                heads = self.head,
                add_self_loops = False,
                concat = True
            ))
            bn_in = self.head*self.size_hidden_layers

        if self.bn_bool == True:
            self.bn = nn.ModuleList()
            self.bn.append(nng.BatchNorm(
                in_channels = bn_in,
                track_running_stats = True,
                allow_single_element = True
            ))
        else:
            self.bn = None


        for n in range(1, self.L):
            if self.pool_type != 'random':
                self.pool.append(nng.TopKPooling(
                    in_channels = self.size_hidden_layers,
                    ratio = self.pool_ratio[n - 1],
                    nonlinearity = torch.sigmoid
                ))

            if self.layer == 'SAGE':
                self.down_layers.append(nng.SAGEConv(
                    in_channels = self.size_hidden_layers,
                    out_channels = 2*self.size_hidden_layers,
                ))
                self.size_hidden_layers = 2*self.size_hidden_layers
                bn_in = self.size_hidden_layers

            elif self.layer == 'GAT':
                self.down_layers.append(nng.GATConv(
                    in_channels = self.head*self.size_hidden_layers,
                    out_channels = self.size_hidden_layers,
                    heads = 2,
                    add_self_loops = False,
                    concat = True
                ))

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = bn_in,
                    track_running_stats = True,
                    allow_single_element = True
                ))

        self.up_layers = nn.ModuleList()

        if self.layer == 'SAGE':
            self.up_layers.append(nng.SAGEConv(
                in_channels = 3*self.size_hidden_layers_init,
                out_channels = self.dim_enc
            ))
            self.size_hidden_layers_init = 2*self.size_hidden_layers_init

        elif self.layer == 'GAT':
            self.up_layers.append(nng.GATConv(
                in_channels = 2*self.head*self.size_hidden_layers,
                out_channels = self.dim_enc,
                heads = 2,
                add_self_loops = False,
                concat = False
            ))

        if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = self.dim_enc,
                    track_running_stats = True,
                    allow_single_element = True
                ))

        for n in range(1, self.L - 1):
            if self.layer == 'SAGE':
                self.up_layers.append(nng.SAGEConv(
                    in_channels = 3*self.size_hidden_layers_init,
                    out_channels = self.size_hidden_layers_init,
                ))
                bn_in = self.size_hidden_layers_init
                self.size_hidden_layers_init = 2*self.size_hidden_layers_init                

            elif self.layer == 'GAT':
                self.up_layers.append(nng.GATConv(
                    in_channels = 2*self.head*self.size_hidden_layers,
                    out_channels = self.size_hidden_layers,
                    heads = 2,
                    add_self_loops = False,
                    concat = True
                ))

            if self.bn_bool == True:
                self.bn.append(nng.BatchNorm(
                    in_channels = bn_in,
                    track_running_stats = True,
                    allow_single_element = True
                ))

    def forward(self, a, input_geom, **kwargs):
        if 'batch' not in kwargs:
            kwargs['batch'] = torch.zeros(
                a.size(0), dtype=torch.long, device=a.device)
        if 'ptr' not in kwargs:
            kwargs['ptr'] = torch.tensor(
                [0, a.size(0)], dtype=torch.long, device=a.device)
        id = []
        batches = [kwargs['batch']]
        ptrs = [kwargs['ptr']]
        if self.training:
            edge_index = nng.radius_graph(
                x = input_geom,
                r = self.r_first,
                loop = True,
                max_num_neighbors = 32,
                batch=kwargs['batch'])
        else:
            edge_index = nng.radius_graph(
                x = input_geom,
                r = self.r_first,
                loop = True,
                max_num_neighbors = 128,
                batch=kwargs['batch'])
        edge_index_list = [edge_index]
        pos_x_list = []
        z = self.encoder(a)
        if self.res:
            z_res = z.clone()

        z = self.down_layers[0](z, edge_index)

        if self.bn_bool == True:
            z = self.bn[0](z)

        z = self.activation(z)
        z_list = [z]
        for n in range(self.L - 1):
            pos_x = input_geom if n == 0 else pos_x[id[n - 1]]
            pos_x_list.append(pos_x)

            ploc_bool = a[:, 0:1] if n == 0 else a[:, 0:1][id[n - 1]]

            if self.pool_type != 'random':
                z, edge_index = DownSample(
                    id,
                    z,
                    edge_index,
                    pos_x,
                    self.pool[n],
                    self.pool_ratio[n],
                    self.list_r[n],
                    self.training,
                    batches,
                    ptrs,
                    ploc_bool)
            else:
                z, edge_index = DownSample(
                    id,
                    z,
                    edge_index,
                    pos_x,
                    None,
                    self.pool_ratio[n],
                    self.list_r[n],
                    self.training,
                    batches,
                    ptrs,
                    ploc_bool)
            edge_index_list.append(edge_index)

            z = self.down_layers[n + 1](z, edge_index)

            if self.bn_bool == True:
                z = self.bn[n + 1](z)

            z = self.activation(z)
            z_list.append(z)
        pos_x_list.append(pos_x[id[-1]])
        
        for n in range(self.L - 1, 0, -1):
            z = UpSample(
                z, pos_x_list[n - 1], pos_x_list[n], batches[n - 1], batches[n])
            z = torch.cat([z, z_list[n - 1]], dim = 1)
            z = self.up_layers[n - 1](z, edge_index_list[n - 1])

            if self.bn_bool == True:
                z = self.bn[self.L + n - 1](z)

            z = self.activation(z) if n != 1 else z

        del(z_list, pos_x_list, edge_index_list, batches, ptrs, id)

        if self.res:
            z = z + z_res

        z = self.decoder(z)

        return z.unsqueeze(1)