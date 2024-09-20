import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from typing import Union

def plot_2D(x, y, title=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if x.dim() == 4:
        cp = ax.contourf(x[0, 0], x[0, 1], y[0, 0], levels=20)
    elif x.dim() == 2:
        cp = ax.tricontourf(x[:, 0], x[:, 1], y[:, 0], levels=20)
    else:
        raise ValueError(f"Unsupported dimension for x.")
    fig.subplots_adjust(right=0.9)
    ax.set_title(title)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(cp, cax=cbar_ax)
    fig.show()

def plot_tri(triang: Triangulation, y: Union[torch.tensor, np.array], title=None):
    plt.figure(figsize=(8, 6))
    tripcolor_plot = plt.tripcolor(triang, y[:, 0], shading='gouraud')
    plt.triplot(triang, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.colorbar(tripcolor_plot)
    plt.show()

def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid. 
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1],
                        height)
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1],
                        width)

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y

class GridEmbedding2D(nn.Module):
    """A simple positional embedding as a regular 2D grid
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbedding2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grid_x, grid_y = regular_grid_2d(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out =  torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)
        # in the unbatched case, the dataloader will stack N 
        # examples with no batch dim to create one
        if not batched and batch_size == 1: 
            return out.squeeze(0)
        else:
            return out