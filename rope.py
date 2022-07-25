import sys
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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


# TODO: IMPLEMENT PHI, THETA, PSI AXES ROTATION FORMALISM
# https://mathworld.wolfram.com/EulerAngles.html
def euler_matrix(phi, theta, psi):
    phi = phi.unsqueeze(-1)#.unsqueeze(-1)
    theta = theta.unsqueeze(-1)#.unsqueeze(-1)
    psi = psi.unsqueeze(-1)#.unsqueeze(-1)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_psi = torch.sin(psi)
    cos_psi = torch.cos(psi)
    a_11 = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi
    a_12 = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi
    a_13 = sin_psi * sin_theta
    a_21 = - sin_psi * cos_phi - cos_theta * sin_phi * cos_psi
    a_22 = - sin_psi * sin_phi + cos_theta * cos_phi * cos_psi
    a_23 = cos_psi * sin_theta
    a_31 = sin_theta * sin_phi
    a_32 = - sin_theta * cos_phi
    a_33 = cos_theta
    a_1_row = torch.cat([a_11, a_12, a_13], dim=-1)
    a_2_row = torch.cat([a_21, a_22, a_23], dim=-1)
    a_3_row = torch.cat([a_31, a_32, a_33], dim=-1)
    a = torch.cat([a_1_row, a_2_row, a_3_row], dim=-2)
    return a



class RopeFinder(nn.Module):
    def __init__(self,
                 initial_point=(200, 200, 60),
                 initial_normal=(1, 1, 0),
                 min_radius=2.,
                 min_height=1.,
                 field_shape=(75, 400, 400),
                 grid_size=6,
                 z_depth=1,
                 kernel_size=1,
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

        self.plane_normal = initial_normal
        # current_normal = self.current_normal()
        self.plane_v, self.plane_w = self.plane_vw(initial_normal)

        # self.plane_v, self.plane_w = self.plane_vw(self.plane_normal)  # INITIALIZE VW --> Vz == 0

        self.phi = nn.Parameter(torch.zeros((1,), dtype=torch.double))
        self.theta = nn.Parameter(torch.zeros((1,), dtype=torch.double))
        self.psi = nn.Parameter(torch.zeros((1,), dtype=torch.double))
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        v, w, n = self.current_vwn()

        self.plane_normal = nn.Parameter(initial_normal, requires_grad=False)
        self.plane_v = nn.Parameter(v, requires_grad=False)
        self.plane_w = nn.Parameter(w, requires_grad=False)

        self.margin_x = self.normalize_point(min_height, self.max_x)
        self.margin_y = self.normalize_point(min_height, self.max_y)
        self.margin_z = self.normalize_point(min_height, self.max_z)

        self.min_radius = min_radius / grid_size
        self.grid_size = grid_size
        self.radius = nn.Parameter(torch.tensor(0.5 * (self.min_radius + 1. / 2.41)))

        initial_point = torch.tensor(initial_point, dtype=torch.double)

        self.o_x = nn.Parameter(self.normalize_point(initial_point[0], self.max_x).double())
        self.o_y = nn.Parameter(self.normalize_point(initial_point[1], self.max_y).double())
        self.o_z = nn.Parameter(self.normalize_point(initial_point[2], self.max_z).double())

        self.z_depth = z_depth

        self.r_phi, (self.v, self.w, self.zz) = self.prepare_regular_grid()
        self.k_e = 50  # FROM DB PAPER https://arxiv.org/pdf/1911.08947.pdf
        self.kernel_size = kernel_size

    def current_normal(self):
        cur_rotation = euler_matrix(self.pi * self.phi, self.pi * self.theta, self.pi * self.psi)
        cn = cur_rotation @ self.plane_normal[:, None]
        cn = cn.T.squeeze(0)
        return cn

    def current_vwn(self):
        cur_rotation = euler_matrix(self.pi * self.phi, self.pi * self.theta, self.pi * self.psi)
        default_vwn = torch.stack([self.plane_v, self.plane_w, self.plane_normal], dim=-1)
        vwn = cur_rotation @ default_vwn
        vwn = vwn.T
        v, w, n = vwn
        return v, w, n

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

    def get_bessel_root_mask(self):
        exp_arg = torch.pow(2.41 * self.radius * self.grid_size - self.r_phi[0], 2) / 2.
        mask = torch.exp(-5. * exp_arg)
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

    def central_kernel(self, f):
        return f[..., self.grid_size - self.kernel_size: self.grid_size + self.kernel_size + 1,
               self.grid_size - self.kernel_size: self.grid_size + self.kernel_size + 1]

    def horizontal_kernel(self, f):
        return f[..., self.grid_size - self.kernel_size: self.grid_size + self.kernel_size + 1, :]

    def vertical_kernel(self, f):
        return f[..., self.grid_size - self.kernel_size: self.grid_size + self.kernel_size + 1]

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

        j_z0 = self.central_kernel(j_z)
        loss_j0 = - torch.mean(j_z0 ** 2)
        sum_j = torch.mean((j_z ** 2).detach()[radial_mask]) + self.eps
        loss_j0 /= sum_j

        b_z0 = self.central_kernel(b_z)
        loss_b0z = - torch.mean(b_z0)
        sum_bz = torch.mean(b_z.abs().detach()[radial_mask]) + self.eps
        loss_b0z /= sum_bz

        loss_br = torch.mean((b_r ** 2)[radial_mask]) / (b_z0 ** 2 + self.eps)
        loss_jr = torch.mean((j_r ** 2)[radial_mask]) / (j_z0 ** 2 + self.eps)

        radial_loss = -torch.mean(radial_mask.double())

        loss = 2 * (loss_b0z + loss_j0) + loss_br + loss_jr + radial_loss

        return loss

    def criterion(self, b, j, b_p, grad_b, grad_grad_b=None):
        """
        Implement here following:
        loss = - Jz|0^2 - Bz|0^2  ############+ (meanBz - Bz_0)^2 + (meanJz - Jz_0)^2 + mean((grad_r Jz)^2) - mean((grad_r Jz|R)^2)
        mean (Br^2) -> 0
        mean (Jr^2) -> 0
        radius -> max
        """
        radial_mask = self.get_radial_mask() #.detach()
        bessel_mask = self.get_bessel_root_mask()

        mask_sum = torch.sum(radial_mask).detach()
        mask_sum_horizontal = torch.sum(self.horizontal_kernel(radial_mask)).detach()
        mask_sum_vertical = torch.sum(self.vertical_kernel(radial_mask)).detach()
        bessel_mask_sum = torch.sum(bessel_mask).detach()

        b_r, b_t, b_z = torch.chunk(b, 3, dim=1)
        j_r, j_t, j_z = torch.chunk(j, 3, dim=1)

        j_z0 = self.central_kernel(j_z)
        loss_j0 = - torch.mean(j_z0 ** 2)
        mean_j = torch.sum((j_z ** 2) * radial_mask) / (mask_sum + self.eps)
        loss_j0 /= mean_j

        b_z0 = self.central_kernel(b_z)
        loss_b0z = - torch.mean(b_z0)
        mean_b = torch.sum(torch.abs(b_z) * radial_mask) / (mask_sum + self.eps)
        loss_b0z /= mean_b

        loss_b_z_bessel = torch.sum(bessel_mask * torch.pow(b_z, 2)) / (bessel_mask_sum + self.eps) / (400 ** 2.)

        loss_br = torch.sum(self.horizontal_kernel((b_r ** 2) * radial_mask)) / (mask_sum_horizontal + self.eps) / 400. ** 2  #/ (b_z0 ** 2 + self.eps)
        loss_jr = torch.sum(self.horizontal_kernel((j_r ** 2) * radial_mask)) / (mask_sum_horizontal + self.eps) / 6000. ** 2  #/ (j_z0 ** 2 + self.eps)

        """
        Grads axial shape:
        B x 3[dR, dTHETA, dZ] x 3[R, THETA, Z] x D x H x W  
        """
        grad_b_r = grad_b[:, :2, 0, ...]
        loss_grad_b_r = torch.sum(self.horizontal_kernel(grad_b_r) ** 2) / (mask_sum_horizontal + self.eps) / 200. ** 2

        if grad_grad_b is not None:
            """
            Grad-grad axial shape:
            B x 3[dR, dTHETA, dZ] x 3[dR, dTHETA, dZ] x 3[R, THETA, Z] x D x H x W 
            """
            loss_grad_grad_b_r = torch.sum(self.vertical_kernel(grad_grad_b[:, 0, 0, 0, ...]) ** 2) / (mask_sum_vertical + self.eps) / 100. ** 2
        else:
            loss_grad_grad_b_r = 0.

        radial_loss = - self.radius

        b_central = self.central_kernel(b_p)  # b_p[..., self.grid_size, self.grid_size]
        b_central = torch.mean(b_central, dim=(-2, -1), keepdim=False)
        b_central = b_central.permute(0, 2, 1)
        b_central = b_central / torch.linalg.norm(b_central, dim=-1, keepdim=True)

        direction_loss = b_central @ self.current_normal().unsqueeze(-1)
        direction_loss = -torch.mean(direction_loss)

        loss = 2 * (loss_b0z + loss_j0) + loss_br + loss_jr + radial_loss +\
               2 * direction_loss + loss_grad_b_r + loss_grad_grad_b_r + loss_b_z_bessel

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
        #current_normal = self.current_normal()
        #plane_v, plane_w = self.plane_vw(current_normal)  #self.plane_normal)
        plane_v, plane_w, current_normal = self.current_vwn()

        grid_v = self.direction_points(self.v, plane_v)
        grid_w = self.direction_points(self.w, plane_w)

        plane_grid = grid_v + grid_w  # 3, H, W, D

        plane_grid = plane_grid.permute(3, 1, 2, 0)  # D, H, W, 3

        plane_grid[..., 0] += self.denormalize_point(self.o_x, self.max_x)
        plane_grid[..., 1] += self.denormalize_point(self.o_y, self.max_y)
        plane_grid[..., 2] += self.denormalize_point(self.o_z, self.max_z)

        plane_vwn = torch.stack([plane_v, plane_w, current_normal], dim=0)

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
        x = F.grid_sample(data, grid_.double().repeat_interleave(b, dim=0), padding_mode='border')
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
        batch, _, _, d, h, w = grads_axial.shape
        grads_axial_reshaped = grads_axial.reshape(batch, 9, 1, d, h, w)
        grads_axial_reshaped = grads_axial_reshaped.squeeze(dim=2)
        grad_grad_f_pol_yx = torch.gradient(grads_axial_reshaped, spacing=1, dim=(-2, -1), edge_order=2)
        grad_grad_axial = self.grad_to_polar(grad_grad_f_pol_yx, polar_matrix)
        grad_grad_axial = grad_grad_axial.unsqueeze(3)
        grad_grad_axial = grad_grad_axial.reshape(batch, 3, 3, 3, d, h, w)
        return f_pol, grads_axial, grad_grad_axial

    def forward(self, f, j=None):
        # DO ALL NORMALIZATIONS BEFORE FORWARD PASS
        with torch.no_grad():
            self.radius.clamp_(self.min_radius, 1. / 2.41)
            self.o_x.clamp_(self.margin_x, -self.margin_x)
            self.o_y.clamp_(self.margin_y, -self.margin_y)
            self.o_z.clamp_(self.margin_z, -self.margin_z)
            self.phi.clamp_(-1, 1)
            self.theta.clamp_(0, 1)
            self.psi.clamp_(-1, 1)

        grid, plane_vwn = self.get_grid()

        if j is not None:
            f = torch.cat([f, j], dim=0)

        f_p_global = self.slice_data(grid, f)
        f_p = self.project_on_plane(f_p_global, plane_vwn)
        f_pol, grad_f_pol, grad_grad_f_pol = self.cylindrical_grad(f_p)
        if j is not None:
            bs = f_p.size(0)
            j_pol = f_pol[bs // 2:]
            f_pol = f_pol[: bs // 2]
            j_p = f_p[bs // 2:]
            f_p = f_p[: bs // 2]
            j_p_global = f_p_global[bs // 2:]
            f_p_global = f_p_global[: bs // 2]

            grad_j_pol = grad_f_pol[bs // 2:]
            grad_f_pol = grad_f_pol[: bs // 2]
            grad_grad_j_pol = grad_grad_f_pol[bs // 2:]
            grad_grad_f_pol = grad_grad_f_pol[: bs // 2]
        else:
            grad_j_pol = None
            j_pol = None
            j_p = None
            grad_grad_j_pol = None
            j_p_global = None

        return f_pol, grad_f_pol, j_pol, grad_j_pol, f_p, j_p, grad_grad_f_pol, grad_grad_j_pol, f_p_global


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


def plot_field(f, name='B', units='G', is_polar=False, z_slice=0, nrows=1, tb_log=True, step=0.4):
    def prepare_labels(field_name, field_units, polar=False):
        if polar:
            lower_names = ('r', r'\tau', 'z')
        else:
            lower_names = ('x', 'y', 'z')
        all_labels = tuple([rf'${field_name}_{ln}, {field_units}$' for ln in lower_names])
        labels_without_unit = tuple([rf'${field_name}_{ln}$' for ln in lower_names])
        return all_labels, labels_without_unit

    labels, labels_without_unit = prepare_labels(field_name=name, field_units=units, polar=is_polar)
    f = torch.flip(f, dims=(-2,))
    f = f.detach().cpu().numpy()
    batch, dim, depth, height, width = f.shape
    x_labels = list(range(- (width // 2), width // 2 + 1))
    x_labels = [step * x for x in x_labels]
    y_labels = list(range(height // 2, - (height // 2 + 1), -1))
    y_labels = [step * x for x in y_labels]
    data = f[0]
    fig, axs = plt.subplots(nrows=nrows, ncols=dim,
                            gridspec_kw=dict(height_ratios=[1], width_ratios=[1] * dim))

    fig.set_size_inches(16, 9)

    for j in range(dim):
        sns.heatmap(data[j, z_slice], linewidth=0.1, ax=axs[j], cmap='vlag', center=0,
                    cbar_kws=dict(use_gridspec=False, location="right", pad=0.01, shrink=min(1., nrows / 2),
                                  label=labels[j])).set(ylabel='y, Mm' if j == 0 else None, xlabel='x, Mm')
        axs[j].set_aspect('equal', 'box')
        axs[j].set_xticks(list(range(0, len(x_labels), 3)))
        axs[j].set_yticks(list(range(0, len(y_labels), 3)))
        axs[j].set_yticklabels([f'{y:.1f}' for y in y_labels[::3]])
        axs[j].set_xticklabels([f'{x:.1f}' for x in x_labels[::3]])

    fig_p, axs_p = plt.subplots(nrows=1, ncols=2, sharey=True,
                                gridspec_kw=dict(height_ratios=[1], width_ratios=[1] * 2))

    fig_p.set_size_inches(16, 9)

    # for k in range(2):
    field = data[:, z_slice, ...]
    field_u = field[:, height // 2 - 1: height // 2 + 2, :]  # TO AVERAGE ALONG DIMENSION
    field_u = np.transpose(field_u, (2, 1, 0))
    field_u = np.reshape(field_u, (-1, 3))
    x_coords = np.array(x_labels).repeat(3, 0)

    df_x = pd.DataFrame(field_u, columns=labels_without_unit, index=x_coords)
    sns.lineplot(data=df_x, ax=axs_p[0], markers=True).set(title=f"u-axis slice", xlabel="x, Mm",
                                                           ylabel=rf"${name}, {units}$")
    axs_p[0].grid()

    field_v = field[..., width // 2 - 1: width // 2 + 2]

    field_v = np.transpose(field_v, (1, 2, 0))
    field_v = np.reshape(field_v, (-1, 3))
    y_coords = np.array(y_labels).repeat(3, 0)

    df_y = pd.DataFrame(field_v, columns=labels_without_unit, index=y_coords)
    sns.lineplot(data=df_y, ax=axs_p[1], markers=True).set(title=f"v-axis slice", xlabel="y, Mm",
                                                           ylabel=rf"${name}, {units}$")
    axs_p[1].grid()

    if tb_log:
        return fig, fig_p
    else:
        plt.savefig('test_fig.png', bbox_inches='tight')


def curl_to_j(curl_val, step_mega_meters=0.4):
    # J = [Field(G)] * c (== 300 000 km/s) / 4Pi / (pixel_size == 400km)
    j_cgse_coeff = (299.792458 / 4. / np.pi / step_mega_meters)
    return curl_val * j_cgse_coeff


def test(file_b=None):
    model = RopeFinder((200, 200, 20))
    b_data = torch.load(file_b).unsqueeze(0)
    j = curl(b_data)
    f_p, grad_f_p, j_p, grad_j_p = model(b_data, j)
    loss = model.criterion(f_p, j_p)
    print(loss)
    # plot_field(j_p)


def main(file_b,
         initial_point=(190., 260., 11.),
         initial_normal=(-1., -1., 0),
         lr=1.e-5,
         max_iterations=2000,
         log_every=2,
         min_height=5,
         grid_size=9,
         radius=3,
         step=0.4):
    b_data = torch.load(file_b).unsqueeze(0)
    bj = curl_to_j(b_data, step_mega_meters=step)
    j = curl(bj)

    writer = SummaryWriter('runs/test')

    initial_normal = (0.5069487516062562, -0.8619013500193842, 0.)

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
        f_p, grad_f_p, j_p, grad_j_p, f_pl, j_pl, grad_grad_f_p, grad_grad_j_p, f_p_global = model(b_data, j)
        loss = model.criterion(f_p, j_p, f_p_global, grad_f_p, grad_grad_f_p)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % log_every == 0:
            model.eval()
            print(f'Iter: {i}. Loss: {running_loss / log_every:.3f}')
            print(f'Initial point: {model.origin}')
            print(f'Plane normal: {model.current_normal().detach().cpu().numpy()}')
            print(f'Radius: {model.r}')
            fig_b, slice_b = plot_field(f_p, 'B', 'G', True, step=step)
            fig_j, slice_j = plot_field(j_p / 1000., 'j', r'10^{3} \cdot statA \cdot cm^{-2}', True, step=step)
            writer.add_figure('B field', fig_b, global_step=i)
            writer.add_figure('B slice', slice_b, global_step=i)
            writer.add_figure('J field', fig_j, global_step=i)
            writer.add_figure('J slice', slice_j, global_step=i)
            writer.add_scalar('Loss', running_loss / log_every, global_step=i)
            running_loss = 0.


if __name__ == '__main__':
    main('b_field.pt')
    #filename = sys.argv[1]
    #save(filename)
