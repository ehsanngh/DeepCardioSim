import torch
import numpy as np
from typing import Union

def determine_batch_size(device, batch_size_base):
    props = torch.cuda.get_device_properties(device)
    total_memory_gb = props.total_memory / (1024**3)  # convert bytes to GB

    # Adjust the batch size based on GPU memory.
    if total_memory_gb < 33:
        batch_size = batch_size_base
    elif total_memory_gb < 42:
        batch_size = int(42 / 33 * batch_size_base)
    elif total_memory_gb < 50:
        batch_size = int(50 / 33 * batch_size_base)
    elif total_memory_gb < 82:
        batch_size = int(82 / 33 * batch_size_base)
    else:
        batch_size = int(140 / 33 * batch_size_base)
    return batch_size


import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

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


def plot_tri(
    triang: Triangulation,
    y: Union[torch.tensor, np.array],
    vmin=None, vmax=None,
    title=None,
    show=True,
    save_dir=None,
    dpi=150):
    
    if vmin is None:
        vmin = y.min()
    if vmax is None:
        vmax = y.max()
    
    plt.figure(figsize=(8, 6))
    tripcolor_plot = plt.tripcolor(
        triang, y[:, 0], shading='gouraud', vmin=vmin, vmax=vmax)
    plt.triplot(triang, color='black', linewidth=0.5)
    
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    
    plt.colorbar(tripcolor_plot)

    if save_dir is not None:
        plt.savefig(save_dir, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close()

    return None

def create_gif_tri(
        triang: Triangulation,
        y: Union[torch.tensor, np.array],
        save_dir: str, title=None, dpi=150):
    import imageio.v3 as iio
    import skimage as ski
    import os
    frames = []
    vmin=y.min()
    vmax=y.max()
    for t in range(y.shape[1]):
        snapshot_num = str(t).zfill(3)
        
        filename = save_dir + snapshot_num +'.png'
        plot_tri(
            triang, y[:, t], show=False,
            save_dir=filename, dpi=dpi, 
            vmin=vmin, vmax=vmax, title=title
            )
        frames.append(ski.io.imread(filename))
        os.remove(filename)

    gif_path = save_dir[:-1] + '.gif'
    frames = np.stack(frames)
    iio.imwrite(gif_path,frames)
    return None