import sys
import torch
from torch import nn
import torch.nn.functional as F

from magnetic.sav2vtk import GXBox


class RopeFinder(nn.Module):
    def __init__(self,
                 initial_point=(200, 200, 60),
                 initial_normal=(1, 1, 0),
                 min_radius=2.,
                 max_radius=9.,
                 min_height=1.,
                 field_shape=(75, 400, 400),
                 grid_size=6,
                 ):
        """
        Min radius is in box relative units.
        Min rope height is in radius units.
        All relative units calculated on z-axis scale.
        Grid size in radius units.
        """
        super(RopeFinder, self).__init__()
        self.field_shape = field_shape
        initial_normal = torch.tensor(initial_normal, dtype=torch.float32)
        plane_normal, plane_v, plane_w = self.plane_vw(initial_normal)
        self.plane_normal = nn.Parameter(plane_normal)
        self.plane_v = nn.Parameter(plane_v, requires_grad=False)
        self.plane_w = nn.Parameter(plane_w, requires_grad=False)
        self.max_radius = max_radius
        self.min_height = min_height

        self.radius = nn.Parameter(torch.tensor(min_radius))
        self.grid_size = grid_size

        initial_point = torch.tensor(initial_point, dtype=torch.float32)
        self.origin = nn.Parameter(initial_point)  # XYZ

    def criterion(self, slice_data):
        """
        Implement here following:
        loss = - Jz_0^2 - Bz_0^2 + (meanBz - Bz_0)^2 + (meanJz - Jz_0)^2 + mean((grad_r Jz)^2) - mean((grad_r Jz|R)^2)
        """
        pass

    # def num_grid_points(self):
    #     return torch.round(self.grid_size * self.radius).int()

    @staticmethod
    def plane_vw(plane_normal):
        """
        Plane equation with PLANE_NORMAL in form: R = R0 + sV + tW
        https://en.wikipedia.org/wiki/Plane_(geometry)
        """
        plane_normal = plane_normal / torch.norm(plane_normal)

        n_x, n_y, n_z = plane_normal
        beta = 1. / torch.sqrt(1. - n_z ** 2)

        v_x = -n_y * beta
        v_y = n_x * beta
        v_z = 0.

        w_x = -n_z * v_y
        w_y = n_z * v_x
        w_z = 1. / beta

        plane_v = torch.tensor([v_x, v_y, v_z], dtype=torch.float32)
        plane_w = torch.tensor([w_x, w_y, w_z], dtype=torch.float32)

        return plane_normal, plane_v, plane_w

    def direction_points(self, grid, direction):
        points = grid[None, ...] * direction.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return points

    def cylindircal_grid(self, v, w):
        v = v[..., -1].squeeze(-1)
        w = w[..., -1].squeeze(-1)
        vw = torch.stack((v, w), dim=-1).double()
        r = torch.norm(vw, dim=-1)
        phi = torch.atan2(vw[..., 1], vw[..., 0])
        return r, phi

    def get_grid(self):
        plane_normal, plane_v, plane_w = self.plane_vw(self.plane_normal)

        grid_xy = torch.arange(-self.grid_size, self.grid_size + 1)
        grid_z = torch.arange(1)

        v, w, zz = torch.meshgrid(grid_xy, grid_xy, grid_z, indexing='xy')

        r, phi = self.cylindircal_grid(v, w)  # SHOULD BE INTERNAL PARAMS

        grid_v = self.direction_points(v, plane_v)
        grid_w = self.direction_points(w, plane_w)

        plane_grid = grid_v + grid_w  # 3, H, W, D

        plane_grid = plane_grid.permute(3, 1, 2, 0)  # D, H, W, 3

        plane_grid += self.origin[None, None, None, :]

        plane_vwn = torch.stack([plane_v, plane_w, plane_normal], dim=0)

        return plane_grid, (r, phi), plane_vwn

    def slice_data(self, grid, data, eps=1e-8):
        if len(grid.shape) == 4:
            grid = grid.unsqueeze(0)
        assert len(grid.shape) == 5
        if len(data.shape) == 4:
            data = data.unsqueeze(0)
        assert len(data.shape) == 5
        dhw = torch.tensor(data.shape[-3:])
        whd = torch.flip(dhw, dims=(-1,))
        scale_factor = (2. / (whd - 1 + eps))
        grid_ = grid * scale_factor[None, None, None, None, :] - 1.
        x = F.grid_sample(data, grid_.double())
        return x

    def cylindrical_grad(self, field, r_phi):
        r, phi = r_phi
        grad_f_yx = torch.gradient(field, spacing=1, dim=(-2, -1), edge_order=2)
        grad_y, grad_x = grad_f_yx
        jacob_x = 1. / torch.cos(phi)
        jacob_x[:, self.grid_size] = 0  # COS_PHI = 0
        jacob_y = 1. / torch.sin(phi)
        jacob_y[self.grid_size, :] = 0  # SIN_PHI = 0
        grad_r = jacob_x[None, None, None, ...] * grad_x + jacob_y[None, None, None, ...] * grad_y
        print(grad_r.shape)
        print(grad_r)
        exit(0)
        print(jacob_y.shape)
        print(jacob_y)
        print(torch.max(torch.abs(jacob_y)))
        print(self.grid_size)


def save(filename):
    box = GXBox(filename)
    b = box.field_to_torch(*box.field)
    j = box.field_to_torch(*box.curl)
    torch.save(b, 'b_field.pt')
    torch.save(j, 'curl.pt')


def test(file_b=None, file_j=None):
    model = RopeFinder((100, 100, 60))
    grid, r_phi, plane_vwn = model.get_grid()
    print(plane_vwn)
    b_data = torch.load(file_b)
    j_data = torch.load(file_j)
    b = model.slice_data(grid, b_data)
    j = model.slice_data(grid, j_data)
    b_c = torch.einsum('bcdhw,nc->bndhw', b, plane_vwn.double())
    print(b_c.shape)

    b_cyl = b.permute(2, 3, 4, 0, 1) @ plane_vwn.T.double()
    b_cyl = b_cyl.permute(3, 4, 0, 1, 2)
    print(b.shape)
    print(b_cyl.shape)
    print(torch.sum(b_cyl - b_c))
    exit(0)
    model.cylindrical_grad(b, r_phi)

if __name__ == '__main__':
    test('b_field.pt', 'curl.pt')
    #filename = sys.argv[1]
    #save(filename)
