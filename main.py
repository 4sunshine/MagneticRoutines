import sys
from tqdm import tqdm

from magnetic.sav2vtk import *
import os
import numpy as np
import cv2
from glob import glob
import json
from magnetic.utils import save_targets_images, plot_data
from magnetic.mathops import curl
import scipy.io as scio
from magnetic.utils import get_times_from_folder, rename_files_accordingly_to_timecodes, main, rename_files_by_timecodes, add_pil_name


def plane_vw(plane_normal):
    """
    Plane equation with PLANE_NORMAL in form: R = R0 + sV + tW
    https://en.wikipedia.org/wiki/Plane_(geometry)
    """
    plane_normal = (plane_normal / np.linalg.norm(plane_normal)).astype(np.float64)

    n_x, n_y, n_z = plane_normal

    beta = 1. / np.sqrt(1. - n_z ** 2)

    v_x = -n_y * beta
    v_y = n_x * beta
    v_z = 0.

    w_x = -n_z * v_y
    w_y = n_z * v_x
    w_z = 1. / beta

    plane_v = np.array([v_x, v_y, v_z], dtype=np.float64)
    plane_w = np.array([w_x, w_y, w_z], dtype=np.float64)

    return plane_normal, plane_v, plane_w


def visualize_data(filename):
    axes = ['r', 'phi', 'z']
    data = np.load(filename, allow_pickle=True)
    # data = np.transpose(data, (1, 0, 2))
    print(data.shape)
    mins = np.min(data, axis=(0, 1), keepdims=True)
    maxs = np.max(data, axis=(0, 1), keepdims=True)
    vis = 255 * (data - mins) / (maxs - mins)
    vis = np.clip(np.round(vis), 0, 255).astype(np.uint8)
    for i in range(3):
        data = np.zeros_like(vis)
        data[..., i] = vis[..., i]
        cv2.imwrite(os.path.join(os.path.dirname(filename),
                                os.path.splitext(os.path.basename(filename))[0] + f'_full_{axes[i]}.png'),
                    data)
    cv2.imwrite(os.path.join(os.path.dirname(filename),
                             os.path.splitext(os.path.basename(filename))[0] + f'_full.png'),
                vis)
    print(data.shape)


def visualize_plot(filename_1, filename_2):
    data_b = np.load(filename_1, allow_pickle=True).item()
    data_j = np.load(filename_2, allow_pickle=True).item()

    bz, d_bz = data_b['b_z']
    b_phi, d_b_phi = data_b['b_phi']
    jz, d_jz = data_j['j_z']
    j_phi, d_j_phi = data_j['j_phi']
    br, d_br = data_b['b_r']
    jr, d_jr = data_j['j_r']
    xs = data_b['radius']
    import matplotlib.pyplot as plt

    plt.axis([0, 6500, -300, 300])
    plt.xlabel(f'r, km; z-height of center = {(15 + 0.52) * 400:.0f} km', fontsize=18)
    plt.ylabel('B, G; j, (Fr/s/cm^2) / 10', fontsize=18)

    plt.plot(xs, bz, marker='o')
    plt.plot(xs[1:], b_phi[1:], marker='o')
    plt.plot(xs[1:], br[1:], marker='o')
    plt.plot(xs, jz / 10, marker='o', linestyle='--')
    plt.plot(xs[1:], j_phi[1:] / 10, marker='o', linestyle='--')
    plt.plot(xs[1:], jr[1:] / 10, marker='o', linestyle='--')
    plt.legend(['B_z', 'B_phi', 'B_r', 'j_z', 'j_phi', 'j_r'], loc='upper right')

    plt.savefig(os.path.join(os.path.dirname(filename_1), 'fields_refined.png'))

    ### CSV GEN
    import csv
    with open(os.path.join(os.path.dirname(filename_1), 'fields_refined.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(('b_z (G)', 'b_phi (G)', 'b_r (G)', 'j_z (Fr/s/cm^2)', 'j_phi (Fr/s/cm^2)', 'j_r (Fr/s/cm^2)', 'r (km)'))
        for i in range(len(bz)):
            w.writerow([bz[i], b_phi[i], br[i], jz[i], j_phi[i], jr[i], xs[i]])

    print(xs)


def visualize_uv_data(filename_1, filename_2):
    data_b = np.load(filename_1, allow_pickle=True)
    data_j = np.load(filename_2, allow_pickle=True)
    assert data_j.shape == data_b.shape
    w, h = data_b.shape[:2]
    ind_x, ind_y = w // 2, h // 2
    # BEAM SIZE 3
    # V+
    r_b = data_b[ind_x + 1:, ind_y - 1: ind_y + 2, :]
    r_j = data_j[ind_x + 1:, ind_y - 1: ind_y + 2, :]
    # V-
    l_b = data_b[:ind_x, ind_y - 1: ind_y + 2, :][::-1, ...]
    l_j = data_j[:ind_x, ind_y - 1: ind_y + 2, :][::-1, ...]
    # U+
    t_b = data_b[ind_x - 1: ind_x + 2, ind_y + 1:, :]
    t_j = data_j[ind_x - 1: ind_x + 2, ind_y + 1:, :]
    # U-
    b_b = data_b[ind_x - 1: ind_x + 2, :ind_y, :][:, ::-1, :]
    b_j = data_j[ind_x - 1: ind_x + 2, :ind_y, :][:, ::-1, :]

    VARS = (r_b, r_j, l_b, l_j, t_b, t_j, b_b, b_j)
    NAMES = ('B_v+', 'j_v+', 'B_v-', 'j_v-', 'B_u+', 'j_u+', 'B_u-', 'j_u-')
    AXIS = (1, 1, 1, 1, 0, 0, 0, 0)
    xs = 400 * np.arange(1, w // 2 + 1)

    import matplotlib.pyplot as plt

    legend = []

    plt.axis([0, 6900, -700, 700])
    plt.xlabel(f'r, km; z-height of center = {(9) * 400:.0f} km', fontsize=18)
    plt.ylabel(r'$B, G; j_{z}, 10^{-1}  statA \cdot cm^{-2}$', fontsize=18)

    coords = ['phi', 'r', 'z']
    result_names = [n[0] + c + n[1:] if i == 0 else 'd_' + n[0] + c + n[1:] for n in NAMES for c in coords for i in range(2)] +\
                   ['r']
    result_table = np.zeros((len(xs), len(result_names)), dtype=np.float32)
    result_table[:, -1] = xs

    for i, (var, name, ax) in enumerate(zip(VARS, NAMES, AXIS)):
        cur_data = np.mean(var, axis=ax)
        cur_std = np.std(var, axis=ax)
        f_phi_r = np.copy(cur_data[:, :2])
        f_z = np.copy(cur_data[:, 2:])
        if i // 2 % 2 == 1:
            f_phi_r = -f_phi_r
        ref_data = np.concatenate([f_phi_r, f_z], axis=-1)
        for j in range(3):
            result_table[:, 6 * i + 2 * j] = ref_data[:, j]
            result_table[:, 6 * i + 2 * j + 1] = cur_std[:, j]


        if i == 0 or i == 4:
            legend.append(name[0] + 'z' + name[1:])
            plt.plot(xs, cur_data[:, 2], marker='o')
            legend.append(name[0] + 'phi' + name[1:])
            plt.plot(xs, cur_data[:, 0], marker='o')
        elif i == 1 or i == 5:
            legend.append(name[0] + 'z' + name[1:])
            plt.plot(xs, cur_data[:, 2] / 10, marker='o', linestyle='--')
            print(name)
        #print(name)
        #print(cur_data)
        #print(cur_std)

    print(result_table)

    plt.legend(legend, loc='upper right')

    plt.savefig(os.path.join(os.path.dirname(filename_1), 'refined_fields_uv.png'))

    ### CSV GEN
    import csv
    with open(os.path.join(os.path.dirname(filename_1), 'refined_fields_uv.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(result_names)
        for i in range(len(xs)):
            w.writerow(result_table[i, :])

    #print(data_b)



if __name__ == '__main__':
    def example():
        from scipy.interpolate import LinearNDInterpolator

        import matplotlib.pyplot as plt

        rng = np.random.default_rng()

        x = np.arange(5)#rng.random(10) + 0.5

        y = (np.arange(5) ** 2) // 10#rng.random(10) - 0.5

        z = x ** 2

        X = np.linspace(min(x), max(x))
        print(X)

        Y = np.linspace(min(y), max(y))

        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        print(X)
        print('***')
        print(X[0, 0])
        print(Y)

        interp = LinearNDInterpolator(list(zip(x, y)), z)

        Z = interp(X, Y)
        print(Z)

        plt.pcolormesh(X, Y, Z, shading='auto')

        plt.plot(x, y, "ok", label="input point")

        plt.legend()

        plt.colorbar()

        plt.axis("equal")

        plt.savefig('example.png')

    #example()
    filename_field = sys.argv[1]
    f2 = sys.argv[2]
    visualize_uv_data(filename_field, f2)
    raise
    # V --> OX, W --> OY, N --> OZ

    def curl_to_j(curl_val):
        # J = [Field(G)] * c (== 300 000 km/s) / 4Pi / (pixel_size == 400km)
        j_cgse_coeff = (300_000 / 4 / np.pi / 400)
        return curl_val * j_cgse_coeff

    filename_csv = sys.argv[2]
    curl, grid = box2curl2grid(filename_field)
    cx, cy, cz = curl
    x, y, z = grid

    from scipy.interpolate import RegularGridInterpolator
    import pandas as pd

    slice_data = pd.read_csv(filename_csv)

    slice_values = np.array(slice_data.values, dtype=np.float64)[:, :3] #J: 3:6
    slice_grid = np.array(slice_data.values, dtype=np.float64)[:, 3:6] #B 3:6

    # PLANE R0, NORMAL
    central_point = np.array([190., 260., 15.], dtype=np.float64)
    plane_normal = np.array([-0.5069487516062562, 0.8619013500193842, 0.], dtype=np.float64)

    centered_grid = slice_grid - central_point[None, :]
    p_n, p_v, p_w = plane_vw(plane_normal)

    v_coords = centered_grid @ p_v[:, None]
    w_coords = centered_grid @ p_w[:, None]
    n_coords = centered_grid @ p_n[:, None]

    # TODO: VWN should be interpolated

    vw_coords = np.concatenate([v_coords, w_coords], axis=1)

    val_v = slice_values @ p_v[:, None]
    val_w = slice_values @ p_w[:, None]
    val_n = slice_values @ p_n[:, None]

    vw_val = np.concatenate([val_v, val_w], axis=1)

    val_r = np.linalg.norm(vw_val, axis=-1, keepdims=True)
    val_phi = np.arctan2(val_w, val_v)

    # VWN == UVW
    val_planar = np.concatenate([val_v, val_w, val_n], axis=1)

    val_cylindrical = np.concatenate([val_r, val_phi, val_n], axis=1)

    CUT_RADIUS = 14

    from scipy.interpolate import LinearNDInterpolator

    regular_points = np.arange(-CUT_RADIUS, CUT_RADIUS + 1)
    V_NEW, W_NEW = np.meshgrid(regular_points, regular_points)
    # interp_r = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 0])
    # interp_phi = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 1])
    # interp_n = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 2])

    interp_v = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_planar[:, 0])
    interp_w = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_planar[:, 1])
    interp_n = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_planar[:, 2])

    V_NEW = V_NEW.astype(np.float64)
    W_NEW = W_NEW.astype(np.float64)

#    R_VAL = interp_r(V_NEW, W_NEW)[..., None].astype(np.float64)
#    PHI_VAL = interp_phi(V_NEW, W_NEW)[..., None].astype(np.float64)
    N_VAL = interp_n(V_NEW, W_NEW).astype(np.float64)
    V_VAL = interp_v(V_NEW, W_NEW).astype(np.float64)
    W_VAL = interp_w(V_NEW, W_NEW).astype(np.float64)

    #print(V_NEW * N_VAL)

    #N_VAL = N_VAL[..., 0]

    # V_C = np.sum(V_NEW * (N_VAL - np.min(N_VAL))) / np.sum((N_VAL - np.min(N_VAL)))
    # W_C = np.sum(W_NEW * (N_VAL - np.min(N_VAL))) / np.sum((N_VAL - np.min(N_VAL)))

    # INDEX B_MIN: [8, 10]
    # V_c, W_c = V_NEW[8, 10], W_NEW[8, 10]
    V_C, W_C = -4., -6.

    #V_C, W_C = 2.7567683836490704, 0.5224803310849225  # CUT RADIUS 14


    # RE-INTERPOLATE AFTER FIRST ESTIMATION

    V_NEW = V_NEW.astype(np.float64) + V_C
    W_NEW = W_NEW.astype(np.float64) + W_C

    V_VAL = interp_v(V_NEW, W_NEW)[..., None].astype(np.float64)
    W_VAL = interp_w(V_NEW, W_NEW)[..., None].astype(np.float64)
    N_VAL = interp_n(V_NEW, W_NEW)[..., None].astype(np.float64)

    result = np.concatenate([V_VAL, W_VAL, N_VAL], axis=-1)

    # MAKE NEW CENTER AN ORIGIN
    V_NEW = V_NEW - V_C
    W_NEW = W_NEW - W_C
    VW = np.concatenate([V_NEW[..., None], W_NEW[..., None]], axis=-1)

    VW_r = np.linalg.norm(VW, axis=-1, keepdims=False)
    # e_phi(tau) = (-sin_phi, cos_phi)
    # e_normal = (cos_phi, sin_phi)
    VW_cos_phi = V_NEW / VW_r
    VW_sin_phi = W_NEW / VW_r

    # BUILD TAU PROJECTION VECTORS FIELD
    VW_TAU = np.stack([-VW_sin_phi, VW_cos_phi], axis=-1)
    VW_RADIAL = np.stack([VW_cos_phi, VW_sin_phi], axis=-1)  # NORMAL/RADIAL DIRECTION

    VAL_VW = np.concatenate([V_VAL, W_VAL], axis=-1)
    print(VAL_VW)

    # CALCULATE
    VAL_TAU = np.nan_to_num(np.sum(VW_TAU * VAL_VW, axis=-1))
    VAL_RADIAL = np.nan_to_num(np.sum(VW_RADIAL * VAL_VW, axis=-1))

    import matplotlib.pyplot as plt

    print(W_NEW + central_point[2] + W_C)

    PIX_SIZE = 400

    plot_W = PIX_SIZE * (W_NEW + central_point[2] + W_C)
    plot_V = PIX_SIZE * (V_NEW)
    inds = np.where(plot_W >= 0)
    rope_center_x, rope_center_y =  plot_V[CUT_RADIUS, CUT_RADIUS], plot_W[CUT_RADIUS, CUT_RADIUS]
    height = int(central_point[2] + W_C)
    cut = CUT_RADIUS - height

    print(np.min(VAL_TAU[cut:]))
    print(np.min(N_VAL[cut:]))
    print(np.min(VAL_RADIAL[cut:]))

    print(np.max(VAL_TAU[cut:]))
    print(np.max(N_VAL[cut:]))
    print(np.max(VAL_RADIAL[cut:]))

    plt.pcolormesh(plot_V[cut:], plot_W[cut:], VAL_TAU[cut:], shading='auto', cmap='seismic',
                   vmin=-500, vmax=500)

    plt.plot(rope_center_x, rope_center_y, "ok", label="flux rope center")

    plt.legend()

    from matplotlib import cm

    #plt.colorbar(label=r'$j_{\varphi}, 10^{3}  statA \cdot cm^{-2}$')
    plt.colorbar(label=r'$B_{\varphi}, G$')

    #plt.colorbar(label=r'$B_{z}, G$')

    plt.xlabel(r'$x, \mathit{km}$', fontsize=18)
    plt.ylabel(r'$z, km$', fontsize=18)

    plt.axis("equal")

    plt.savefig(os.path.join(os.path.dirname(filename_csv),
                         os.path.splitext(os.path.basename(filename_csv))[0] + '_phi.png'))

    VALUE = np.stack([VAL_TAU, VAL_RADIAL, N_VAL[..., 0]], axis=-1)#[cut:]

    data = VALUE

    np.save(os.path.join(os.path.dirname(filename_csv),
                         os.path.splitext(os.path.basename(filename_csv))[0] + '_refined_data_uv_plane.npy'),
            data, allow_pickle=True)

    raise

    fn = []
    d_fn = []
    fphi = []
    d_fphi = []
    fr = []
    d_fr = []
    radius = []
    # PIXEL_SCALE = 400 km
    PX_SCALE = 400
    for i in range(CUT_RADIUS):
        radius.append(PX_SCALE * i)
        r_cur = np.where((VW_r >= i) & (VW_r < (i + 1)))
        z_cur = N_VAL[r_cur]
        phi_cur = VAL_TAU[r_cur]
        radial_cur = VAL_RADIAL[r_cur]
        fn.append(np.mean(z_cur))
        d_fn.append(np.std(z_cur))
        fphi.append(np.mean(phi_cur))
        d_fphi.append(np.std(phi_cur))
        fr.append(np.mean(radial_cur))
        d_fr.append(np.std(radial_cur))

    data = {'j_phi': [curl_to_j(np.array(fphi)), curl_to_j(np.array(d_fphi))],
            'j_z': [curl_to_j(np.array(fn)), curl_to_j(np.array(d_fn))],
            'j_r': [curl_to_j(np.array(fr)), curl_to_j(np.array(d_fr))],
            'radius': np.array(radius)}

    np.save(os.path.join(os.path.dirname(filename_csv),
                         os.path.splitext(os.path.basename(filename_csv))[0] + '_data_cut_uv.npy'),
            data, allow_pickle=True)

    #main()
    # base_dir = sys.argv[1]
    # subdirs = sorted(os.listdir(base_dir))
    # subdirs = [s for s in subdirs if os.path.isdir(os.path.join(base_dir, s))]
    # for folder in tqdm(subdirs, total=len(subdirs)):
    #     current_base_dir = os.path.join(base_dir, folder)
    #     time_file = os.path.join(current_base_dir, 'timecodes.json')
    #     cur_subdirs = sorted(os.listdir(current_base_dir))
    #     cur_subdirs = [s for s in cur_subdirs if os.path.isdir(os.path.join(current_base_dir, s))]
    #     for cs in cur_subdirs:
    #         add_pil_name(os.path.join(current_base_dir, cs))
    #         # rename_files_by_timecodes(time_file, )


    #     convert_folder_serial(os.path.join(base_dir, folder), 'curl_B', box2curl2vtk)
    # convert_folder('/media/sunshine/HDD/thz_events/20120704/NLFFFE_sav', 'jB_closure', func=box2directions2vtk, n_jobs=8)
    # convert_folder('/media/sunshine/HDD/solar_data/BASEMAPS/Vert_currenthmi.M_720s.20120704_095826.E145S20CR.CEA.BND.vtk')
    # subfolders = glob('/media/sunshine/HDD/thz_events/20120704/*/')
    # rename_files_accordingly_to_timecodes('/media/sunshine/HDD/thz_events/20120704/timecodes.json', subfolders)
    #get_times_from_folder('/media/sunshine/HDD/thz_events/20120704/NLFFFE_sav')
    #get_image2_from_sav('/media/sunshine/HDD/thz_events/20120704/BASEMAPS/AIA_94NLFFFE_120703_213425.sav')
    #box2vtk('/media/sunshine/HDD/thz_events/20120704/NLFFFE_sav/NLFFFE_120704_164625.sav', 'B_nlfffe')
    #convert_folder('/media/sunshine/HDD/thz_events/20120704', 'curl_B', func=box2curl2vtk, n_jobs=6, last=6)


    # LINE = 131
    # target_dir = f'/media/sunshine/HDD/Loops/target_loops_circles_morning_deepcopy_{LINE}'
    # save_targets_images(target_dir, LINE)
    #
    # CURL_TO_AMPERE = 3 * 1.e9 / np.pi
    #
    # begin_currents = np.load(os.path.join(target_dir, f'begin_currents_{LINE}.npy'))
    # end_currents = np.load(os.path.join(target_dir, f'end_currents_{LINE}.npy'))
    # print(begin_currents[:, 1])
    # print(end_currents[:, 1])
    # print(begin_currents[:, 1] - end_currents[:, 1])
    # for i in range(3):
    #     data = np.array([begin_currents[:, i], end_currents[:, i], begin_currents[:, i] - end_currents[:, i]])
    #     data *= CURL_TO_AMPERE
    #     plot_data(data, os.path.join(target_dir, f'save_{i}.png'))
    # save_targets_images()
    # print('Hello')
    # looptrace = '/media/sunshine/HDD/Loops/loops_final_94/traces_AIA_94NORH_NLFFFE_170904_055842.dat'
    # loops, ends = read_looptrace(looptrace)
    # signals = []
    # mean_signal = {}
    # for k, v in loops.items():
    #     signal = np.array(v['signal'])
    #     signals.append(np.mean(signal))
    #     mean_signal[k] = np.mean(signal)
    #
    # mean_signal = dict(sorted(mean_signal.items(), key=lambda item: item[1], reverse=True))
    # # TOP 3 BRIGHTEST LOOPS
    # for i, k in enumerate(mean_signal.keys()):
    #     coords = loops[k]['points']
    #     if i == 2:
    #         break
    # print(signals)
    # print(mean_signal)



    # data = np.load('/media/sunshine/HDD/nasa_dataset/hmi_bz/HMI_Bz_201709/09/01/HMI20170901_0000_bz.npz')
    # data = np.load('/media/sunshine/HDD/nasa_dataset/aia_094/AIA_0094_201709/09/01/AIA20170901_0000_0094.npz')
    # x = data['x']
    # print(np.min(x))
    # print(np.max(x))
    # new_x = np.round(255 * (x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.uint8)
    # print(np.shape(new_x))
    # print(np.min(new_x))
    # print(np.max(new_x))
    # new_x = cv2.cvtColor(new_x, cv2.COLOR_GRAY2BGR)
    # print(np.shape(new_x))
    # cv2.imshow('HMI', new_x)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # folder = 'C:/AppsData/gx_out'
    # folder_derivative(folder, 'dB', filter='.vtk')
    # convert_folder(folder, 'Bpot', func=box2npy)
    # free_energy('C:\\AppsData\\gx_out\\npy_field', 'C:\\AppsData\\NLFFFE\\npy_field')
    # looptrace = 'C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat'
    # save_file = os.path.join(os.path.dirname(looptrace), 'test_sources')
    # source_points(looptrace, save_file)
    # regular_grid('/home/sunshine/regular')
    #read_looptrace('C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat')
