import sys
import torch
from torch import nn
import torch.nn.functional as F

from magnetic.sav2vtk import GXBox


def levi_civita_3d():
    e = torch.zeros((3, 3, 3), dtype=torch.double)
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[2, 1, 0] = e[0, 2, 1] = e[1, 0, 2] = -1
    return e


def curl(f, spacing=1, edge_order=2):
    assert 4 <= len(f.shape) <= 5
    if len(f.shape) == 4:
        f = f.unsqueeze(0)
    grad_f_zyx = torch.gradient(f, spacing=spacing, dim=(-3, -2, -1), edge_order=edge_order)
    grad_z, grad_y, grad_x = grad_f_zyx
    grads = torch.stack([grad_x, grad_y, grad_z], dim=1)
    lc = levi_civita_3d()
    curl = torch.einsum('ijk,bjkdhw->bidhw', lc, grads)
    return curl


class RopeFinder(nn.Module):
    def __init__(self,
                 initial_point=(200, 200, 60),
                 initial_normal=(1, 1, 0),
                 min_radius=2.,
                 max_radius=9.,
                 min_height=1.,
                 field_shape=(75, 400, 400),
                 grid_size=6,
                 z_depth=1,
                 ):
        """
        Min radius is in box relative units.
        Min rope height is in radius units.
        All relative units calculated on z-axis scale.
        Grid size in radius units.
        All diff [dxF] tensor orders: d_i x F_j
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
        self.z_depth = z_depth

        self.r_phi, (self.v, self.w, self.zz) = self.prepare_regular_grid()

    def criterion(self, slice_data):
        """
        Implement here following:
        loss = - Jz|0^2 - Bz|0^2  ############+ (meanBz - Bz_0)^2 + (meanJz - Jz_0)^2 + mean((grad_r Jz)^2) - mean((grad_r Jz|R)^2)
        mean (Br^2) -> 0
        mean (Jr^2) -> 0
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

    def prepare_regular_grid(self):
        grid_xy = torch.arange(-self.grid_size, self.grid_size + 1)
        grid_z = torch.arange(self.z_depth)

        v, w, zz = torch.meshgrid(grid_xy, grid_xy, grid_z, indexing='xy')

        r, phi = self.cylindircal_grid(v, w)  # SHOULD BE INTERNAL PARAMS
        return (r, phi), (v, w, zz)

    def cylindircal_grid(self, v, w):
        v = v[..., -1].squeeze(-1)
        w = w[..., -1].squeeze(-1)
        vw = torch.stack((v, w), dim=-1).double()
        r = torch.norm(vw, dim=-1)
        phi = torch.atan2(vw[..., 1], vw[..., 0])
        return r, phi

    def get_grid(self):
        plane_normal, plane_v, plane_w = self.plane_vw(self.plane_normal)

        grid_v = self.direction_points(self.v, plane_v)
        grid_w = self.direction_points(self.w, plane_w)

        plane_grid = grid_v + grid_w  # 3, H, W, D

        plane_grid = plane_grid.permute(3, 1, 2, 0)  # D, H, W, 3

        plane_grid += self.origin[None, None, None, :]

        plane_vwn = torch.stack([plane_v, plane_w, plane_normal], dim=0)

        return plane_grid, plane_vwn.double()

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

    def get_polar_matrix(self, r_phi):
        r, phi = r_phi
        """https://www2.physics.ox.ac.uk/sites/default/files/2011-10-08/coordinates_pdf_51202.pdf"""
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        e_r = torch.stack((cos_phi, sin_phi, torch.zeros_like(cos_phi)), dim=-1)  #.unsqueeze(0)  # HWC:2
        e_phi = torch.stack((-sin_phi, cos_phi, torch.zeros_like(cos_phi)), dim=-1)  #.unsqueeze(0)
        e_n = torch.zeros_like(e_phi)
        e_n[:, :, -1] = 1
        polar_matrix = torch.stack((e_r, e_phi, e_n), dim=0)
        return polar_matrix

    def cartesian_to_polar(self, field, polar_matrix):
        field_axial = torch.einsum('bcdhw,nhwc->bndhw', field, polar_matrix)  # B3DHW, 3: R, PHI, N
        return field_axial

    def grad_to_polar(self, f, polar_matrix):
        assert 2 <= len(f) <= 3
        grad_y, grad_x = f[:2]
        if len(f) == 2:
            grad_z = torch.zeros_like(f[-1])
        else:
            grad_z = f[2]
        grads = torch.stack((grad_x, grad_y, grad_z), dim=1)
        grads_axial = torch.einsum('bcidhw,nhwc->bnidhw', grads, polar_matrix)  # B3DHW, 3: R, PHI, N
        """
        Grads axial shape:
        B x 3[dR, dTHETA, dZ] x 3[R, THETA, Z] x D x H x W  
        """
        return grads_axial

    def project_on_plane(self, f, plane_vwn):
        return torch.einsum('bcdhw,nc->bndhw', f, plane_vwn)

    def cylindrical_grad(self, field):
        polar_matrix = self.get_polar_matrix(self.r_phi)
        f_pol = self.cartesian_to_polar(field, polar_matrix)
        grad_f_pol_yx = torch.gradient(f_pol, spacing=1, dim=(-2, -1), edge_order=2)
        grads_axial = self.grad_to_polar(grad_f_pol_yx, polar_matrix)
        return grads_axial

    def forward(self, f):
        grid, plane_vwn = self.get_grid()
        f_p = self.slice_data(grid, f)
        f_p = self.project_on_plane(f_p, plane_vwn)
        grad_f_p = self.cylindrical_grad(f_p)
        return f_p, grad_f_p


def save(filename):
    box = GXBox(filename)
    b = box.field_to_torch(*box.field)
    j = box.field_to_torch(*box.curl)
    torch.save(b, 'b_field.pt')
    torch.save(j, 'curl.pt')


def test(file_b=None):
    model = RopeFinder((100, 100, 60))
    b_data = torch.load(file_b)
    f_p, grad_f_p = model(b_data)
    j = curl(b_data)


if __name__ == '__main__':
    test('b_field.pt')
    #filename = sys.argv[1]
    #save(filename)
