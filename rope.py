import sys
import torch
from torch import nn
import torch.nn.functional as F

from magnetic.sav2vtk import GXBox


class RopeFinder(nn.Module):
    def __init__(self,
                 initial_point=(20, 20, 35),
                 initial_normal=(1, 1, 0),
                 min_radius=2.,
                 max_radius=20.,
                 min_height=1.,
                 field_shape=(75, 400, 400),
                 grid_size=3,
                 ):
        """
        Min radius is in box relative units.
        Min rope height is in radius units.
        All relative units calculated on z-axis scale.
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
        self.grid_res_z = 2. / self.field_shape[0]
        assert self.field_shape[1] == self.field_shape[2]
        self.grid_res_xy = 2. / self.field_shape[1]
        self.radius_z = nn.Parameter(torch.tensor(min_radius * self.grid_res_z))
        self.radius_xy = nn.Parameter(torch.tensor(min_radius * self.grid_res_xy))
        self.grid_size = grid_size

        initial_point = torch.tensor(initial_point, dtype=torch.float32)
        initial_point[:2] *= self.grid_res_xy
        initial_point[2] *= self.grid_res_z
        self.origin = nn.Parameter(initial_point)

    def criterion(self, slice_data):
        pass

    def num_grid_points(self):
        return self.grid_size * self.radius_z / self.grid_res_z

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
        points[:2, ...] *= self.grid_res_xy
        points[2, ...] *= self.grid_res_z
        return points

    def get_grid(self):
        plane_normal, plane_v, plane_w = self.plane_vw(self.plane_normal)

        num_points = self.num_grid_points()
        grid_xy = torch.arange(-num_points.item(), num_points.item() + 1)
        grid_z = torch.arange(1.)

        v, w, zz = torch.meshgrid(grid_xy, grid_xy, grid_z, indexing='ij')
        grid_v = self.direction_points(v, plane_v)
        grid_w = self.direction_points(w, plane_w)

        plane_grid = grid_v + grid_w
        plane_grid = plane_grid.permute(3, 2, 1, 0)
        plane_grid += self.origin[None, None, None, :]

        return plane_grid


def save(filename):
    box = GXBox(filename)
    b = box.field_to_torch(*box.field)
    j = box.field_to_torch(*box.curl)
    torch.save(b, 'b_field.pt')
    torch.save(j, 'curl.pt')


def test(filename=None):
    model = RopeFinder()
    grid = model.get_grid()
    data = torch.load(filename)
    data = data.unsqueeze(0)
    grid = grid.unsqueeze(0)
    x = F.grid_sample(data, grid.double())
    x = torch.norm(x, dim=1)[0, 0]
    print(x.shape)
    print(x)


if __name__ == '__main__':
    test('b_field.pt')
    #filename = sys.argv[1]
    #save(filename)
