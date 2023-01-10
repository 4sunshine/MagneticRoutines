import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


def bottom_slice(data):
    assert len(data.shape) > 2
    return data[..., 0, :, :]


def plot_field(data, axis=-1):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data[axis, :, :]
    h, w = data.shape
    max_val = np.abs(data).max()
    min_val = -max_val
    fig, ax = plt.subplots()
    c = ax.pcolormesh(data, cmap='RdBu', vmin=min_val, vmax=max_val)
    ax.axis([0, w, 0, h])
    fig.colorbar(c, ax=ax)
    plt.savefig('output/bottom.jpg')


def create_meshgrid(min_x, max_x, min_y, max_y, min_z, max_z):
    x = torch.arange(min_x, max_x)
    y = torch.arange(min_y, max_y)
    z = torch.arange(min_z, max_z)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    all_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    all_coords = torch.flatten(all_coords, start_dim=0, end_dim=-2)
    return all_coords


class BOXDataset(Dataset):
    """
    Data shape is: [C, D, H, W]
    dim = 3
    depth = D. D = 0 is bottom
    height = H: H ^|^| OY
    width = W: W ^|^| OX
    """
    def __init__(self, data_file, transform=None, train_val_mask=None):
        self.data = torch.load(data_file)
        self.transform = transform
        self.bottom_boundary = bottom_slice(self.data)
        self.z_max, self.y_max, self.x_max = self.data.shape[-3:]
        self.all_coords = create_meshgrid(0, self.x_max, 0, self.y_max, 0, self.z_max).long()  # all X,Y,Z's in cube
        self.train_val_mask = train_val_mask

    def __len__(self):
        return len(self.all_coords)

    def __getitem__(self, idx):
        coord = self.all_coords[idx]
        x, y, z = coord
        current_boundary = self.bottom_boundary[:, y, x]
        gt_field = self.data[:, z, y, x]
        coord = coord.float()
        boundary_coord = coord.clone()
        boundary_coord[-1] = 0.
        return coord, gt_field, boundary_coord, current_boundary


if __name__ == '__main__':
    ds = BOXDataset(sys.argv[1])
    for d in ds:
        print(d)
        break
