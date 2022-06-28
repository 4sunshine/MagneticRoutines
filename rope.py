import sys
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from tensorboardX import SummaryWriter

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
        self.max_z, self.max_y, self.max_x = field_shape
        self.whd = torch.tensor([self.max_x, self.max_y, self.max_z], dtype=torch.double)
        self.eps = 1e-8

        initial_normal = torch.tensor(initial_normal, dtype=torch.double)
        initial_normal /= torch.linalg.norm(initial_normal)

        plane_v, plane_w = self.plane_vw(initial_normal)
        self.plane_normal = nn.Parameter(initial_normal)

        self.plane_v = nn.Parameter(plane_v, requires_grad=False)
        self.plane_w = nn.Parameter(plane_w, requires_grad=False)

        self.margin_x = self.normalize_point(min_height, self.max_x)
        self.margin_y = self.normalize_point(min_height, self.max_y)
        self.margin_z = self.normalize_point(min_height, self.max_z)

        self.min_radius = min_radius / grid_size
        self.grid_size = grid_size
        self.radius = nn.Parameter(torch.tensor(0.5 * (self.min_radius + 1.)))

        initial_point = torch.tensor(initial_point, dtype=torch.double)

        self.o_x = nn.Parameter(self.normalize_point(initial_point[0], self.max_x).double())
        self.o_y = nn.Parameter(self.normalize_point(initial_point[1], self.max_y).double())
        self.o_z = nn.Parameter(self.normalize_point(initial_point[2], self.max_z).double())

        self.z_depth = z_depth

        self.r_phi, (self.v, self.w, self.zz) = self.prepare_regular_grid()
        self.k_e = 50  # FROM DB PAPER https://arxiv.org/pdf/1911.08947.pdf

    def normalize_point(self, x, max_size):
        factor = torch.tensor(2.) / torch.tensor(max_size - 1.).clamp(self.eps)
        return factor * x - 1

    def denormalize_point(self, x, max_size):
        factor = torch.tensor(2.) / torch.tensor(max_size - 1.).clamp(self.eps)
        return 1. / factor * (x + 1.)

    # def get_radial_mask(self):
    #     mask = self.r_phi[0] <= self.radius
    #     return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def get_radial_mask(self):
        exp_arg = self.k_e * (self.radius * self.grid_size - self.r_phi[0])
        mask = 1. / (1 + torch.exp(-exp_arg))
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    @property
    @torch.no_grad()
    def origin(self):
        ox = self.denormalize_point(self.o_x.detach().cpu().numpy(), self.max_x)
        oy = self.denormalize_point(self.o_y.detach().cpu().numpy(), self.max_y)
        oz = self.denormalize_point(self.o_z.detach().cpu().numpy(), self.max_z)
        return ox.item(), oy.item(), oz.item()

    @property
    @torch.no_grad()
    def r(self):
        radius = self.radius.detach().cpu().numpy() * self.grid_size
        return radius.item()

    def criterion_non_db(self, b, j):
        """
        Implement here following:
        loss = - Jz|0^2 - Bz|0^2  ############+ (meanBz - Bz_0)^2 + (meanJz - Jz_0)^2 + mean((grad_r Jz)^2) - mean((grad_r Jz|R)^2)
        mean (Br^2) -> 0
        mean (Jr^2) -> 0
        radius -> max
        """
        radial_mask = self.get_radial_mask()

        b_r, b_t, b_z = torch.chunk(b, 3, dim=1)
        j_r, j_t, j_z = torch.chunk(j, 3, dim=1)

        j_z0 = j_z[..., self.grid_size, self.grid_size]
        loss_j0 = - torch.mean(j_z0 ** 2)
        sum_j = torch.mean((j_z ** 2).detach()[radial_mask]) + self.eps
        loss_j0 /= sum_j

        b_z0 = b_z[..., self.grid_size, self.grid_size]
        loss_b0z = - torch.mean(b_z0)
        sum_bz = torch.mean(b_z.abs().detach()[radial_mask]) + self.eps
        loss_b0z /= sum_bz

        loss_br = torch.mean((b_r ** 2)[radial_mask]) / (b_z0 ** 2 + self.eps)
        loss_jr = torch.mean((j_r ** 2)[radial_mask]) / (j_z0 ** 2 + self.eps)

        radial_loss = -torch.mean(radial_mask.double())

        loss = 2 * (loss_b0z + loss_j0) + loss_br + loss_jr + radial_loss

        return loss

    def criterion(self, b, j, b_p):
        """
        Implement here following:
        loss = - Jz|0^2 - Bz|0^2  ############+ (meanBz - Bz_0)^2 + (meanJz - Jz_0)^2 + mean((grad_r Jz)^2) - mean((grad_r Jz|R)^2)
        mean (Br^2) -> 0
        mean (Jr^2) -> 0
        radius -> max
        """
        radial_mask = self.get_radial_mask() #.detach()

        mask_sum = torch.sum(radial_mask).detach()

        b_r, b_t, b_z = torch.chunk(b, 3, dim=1)
        j_r, j_t, j_z = torch.chunk(j, 3, dim=1)

        j_z0 = j_z[..., self.grid_size, self.grid_size]
        loss_j0 = - torch.mean(j_z0 ** 2)
        mean_j = torch.sum((j_z ** 2) * radial_mask) / mask_sum + self.eps
        loss_j0 /= mean_j

        b_z0 = b_z[..., self.grid_size, self.grid_size]
        loss_b0z = - torch.mean(b_z0)
        mean_b = torch.sum(torch.abs(b_z) * radial_mask) / mask_sum + self.eps
        loss_b0z /= mean_b

        loss_br = torch.sum((b_r ** 2) * radial_mask) / (b_z0 ** 2 + self.eps) / mask_sum
        loss_jr = torch.sum((j_r ** 2) * radial_mask) / (j_z0 ** 2 + self.eps) / mask_sum

        radial_loss = - self.radius

        b_central = b_p[..., self.grid_size, self.grid_size]
        b_central = b_central.permute(0, 2, 1)
        b_central = b_central / torch.linalg.norm(b_central, dim=-1, keepdim=True)

        plane_normal_norm = torch.linalg.norm(self.plane_normal)
        direction_loss = b_central @ self.plane_normal.unsqueeze(-1) / plane_normal_norm
        direction_loss = -torch.mean(direction_loss)

        loss = 2 * (loss_b0z + loss_j0) + loss_br + loss_jr + radial_loss + 2 * direction_loss

        return loss

    # def num_grid_points(self):
    #     return torch.round(self.grid_size * self.radius).int()

    @staticmethod
    def plane_vw(plane_normal):
        """
        Plane equation with PLANE_NORMAL in form: R = R0 + sV + tW
        https://en.wikipedia.org/wiki/Plane_(geometry)
        """
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

        return plane_v, plane_w

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
        r = torch.linalg.norm(vw, dim=-1)
        phi = torch.atan2(vw[..., 1], vw[..., 0])
        return r, phi

    def get_grid(self):
        plane_v, plane_w = self.plane_vw(self.plane_normal)

        grid_v = self.direction_points(self.v, plane_v)
        grid_w = self.direction_points(self.w, plane_w)

        plane_grid = grid_v + grid_w  # 3, H, W, D

        plane_grid = plane_grid.permute(3, 1, 2, 0)  # D, H, W, 3

        plane_grid[..., 0] += self.denormalize_point(self.o_x, self.max_x)
        plane_grid[..., 1] += self.denormalize_point(self.o_y, self.max_y)
        plane_grid[..., 2] += self.denormalize_point(self.o_z, self.max_z)

        plane_vwn = torch.stack([plane_v, plane_w, self.plane_normal], dim=0)

        return plane_grid, plane_vwn.double()

    def slice_data(self, grid, data):
        if len(grid.shape) == 4:
            grid = grid.unsqueeze(0)
        assert len(grid.shape) == 5
        if len(data.shape) == 4:
            data = data.unsqueeze(0)
        assert len(data.shape) == 5
        dhw = torch.tensor(data.shape[-3:])
        whd = torch.flip(dhw, dims=(-1,))
        scale_factor = (2. / (whd - 1 + self.eps))
        grid_ = grid * scale_factor[None, None, None, None, :] - 1.

        b = data.shape[0]
        x = F.grid_sample(data, grid_.double().repeat_interleave(b, dim=0))
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
        return f_pol, grads_axial

    def forward(self, f, j=None):
        # DO ALL NORMALIZATIONS BEFORE FORWARD PASS
        with torch.no_grad():
            self.radius.clamp_(self.min_radius, 1.)
            self.o_x.clamp_(self.margin_x, -self.margin_x)
            self.o_y.clamp_(self.margin_y, -self.margin_y)
            self.o_z.clamp_(self.margin_z, -self.margin_z)
            self.plane_normal.clamp_(-1, 1)
            self.plane_normal.div_(self.plane_normal.norm())

        grid, plane_vwn = self.get_grid()

        if j is not None:
            f = torch.cat([f, j], dim=0)

        f_p = self.slice_data(grid, f)
        f_p = self.project_on_plane(f_p, plane_vwn)
        f_pol, grad_f_pol = self.cylindrical_grad(f_p)
        if j is not None:
            bs = f_p.size(0)
            j_pol = f_pol[bs // 2:]
            f_pol = f_pol[: bs // 2]
            j_p = f_p[bs // 2:]
            f_p = f_p[: bs // 2]

            grad_j_pol = grad_f_pol[bs // 2:]
            grad_f_pol = grad_f_pol[: bs // 2]
        else:
            grad_j_pol = None
            j_pol = None
            j_p = None

        return f_pol, grad_f_pol, j_pol, grad_j_pol, f_p, j_p


def save(filename):
    box = GXBox(filename)
    b = box.field_to_torch(*box.field)
    j = box.field_to_torch(*box.curl)
    torch.save(b, 'b_field.pt')
    torch.save(j, 'curl.pt')


def save_points(data, plane_n, grid_size):
    import numpy as np
    from pyevtk.hl import pointsToVTK
    data = data[0]
    points_y_plus = data[grid_size:, grid_size, :].T.detach().numpy()
    points_x_plus = data[grid_size, grid_size:, :].T.detach().numpy()
    points_d = torch.diagonal(data)[:, grid_size:].detach().numpy()
    xd, yd, zd = points_d
    xt, yt, zt = points_y_plus
    xp, yp, zp = points_x_plus
    origin = points_x_plus[:, 0]
    points_n = torch.arange(1, 10)[:, None]
    plane_n_ = plane_n[None, :] * points_n
    plane_n_ = plane_n_.T
    plane_n_ = plane_n_.detach().numpy()
    plane_n_ = plane_n_ + origin[:, None]
    px, py, pz = plane_n_
    x_points = np.concatenate([xd, xt, xp, px]).flatten()
    y_points = np.concatenate([yd, yt, yp, py]).flatten()
    z_points = np.concatenate([zd, zt, zp, pz]).flatten()
    test_data = np.ones(np.shape(x_points)[0])
    pointsToVTK('plane_vtk', x_points, y_points, z_points, {'source': test_data})


def plot_field(f, labels=(r'$B_r, G$', r'$B_\tau, G$', r'$B_z, G$'), z_slice=0, nrows=1, tb_log=True):
    f = torch.flip(f, dims=(-2,))
    f = f.detach().cpu().numpy()
    batch, dim, depth, height, width = f.shape
    data = f[0]
    fig, axs = plt.subplots(nrows=nrows, ncols=dim, sharex=True,
                            gridspec_kw=dict(height_ratios=[1], width_ratios=[1] * dim))

    fig.set_size_inches(16, 9)

    for j in range(dim):
        sns.heatmap(data[j, z_slice], linewidth=0.1, ax=axs[j],
                    cbar_kws=dict(use_gridspec=False, location="right", pad=0.01, shrink=min(1., nrows / 2),
                                  label=labels[j]))
        axs[j].set_aspect('equal', 'box')
        axs[j].set_yticklabels(list(range(height // 2, - (height // 2 + 1), -1)))
        axs[j].set_xticklabels(list(range(- (width // 2), width // 2 + 1)))

    if tb_log:
        return fig
    else:
        plt.savefig('test_fig.png', bbox_inches='tight')


def test(file_b=None):
    model = RopeFinder((200, 200, 20))
    b_data = torch.load(file_b).unsqueeze(0)
    j = curl(b_data)
    f_p, grad_f_p, j_p, grad_j_p = model(b_data, j)
    loss = model.criterion(f_p, j_p)
    print(loss)
    # plot_field(j_p)


def main(file_b,
         initial_point=(200, 250, 20),
         initial_normal=(-1., -1., 0),
         lr=2.e-6,
         max_iterations=2000,
         log_every=2,
         min_height=5,
         grid_size=9,
         radius=6):
    b_data = torch.load(file_b).unsqueeze(0)
    j = curl(b_data)

    writer = SummaryWriter('runs/test')

    model = RopeFinder(initial_point,
                       min_height=min_height,
                       grid_size=grid_size,
                       min_radius=radius,
                       initial_normal=initial_normal)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0

    for i in range(max_iterations):
        model.train()
        optimizer.zero_grad()
        f_p, grad_f_p, j_p, grad_j_p, f_pl, j_pl = model(b_data, j)
        loss = model.criterion(f_p, j_p, f_pl)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % log_every == 0:
            model.eval()
            print(f'Iter: {i}. Loss: {running_loss / log_every:.3f}')
            print(f'Initial point: {model.origin}')
            print(f'Plane normal: {model.plane_normal.detach().cpu().numpy()}')
            print(f'Radius: {model.r}')
            fig = plot_field(f_p)
            writer.add_figure('Field', fig, global_step=i)
            writer.add_scalar('Loss', running_loss / log_every, global_step=i)
            running_loss = 0.


if __name__ == '__main__':
    main('b_field.pt')
    #filename = sys.argv[1]
    #save(filename)
