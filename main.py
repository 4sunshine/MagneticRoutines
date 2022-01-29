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

    print(data_j)
    print(data_b)
    b_z = data_b['b_z']
    b_phi = 180 * data_b['b_phi'] / np.pi
    j_z = data_j['j_z']
    j_phi = 180 * data_j['j_phi'] / np.pi
    xs = np.arange(len(b_z))
    import matplotlib.pyplot as plt

    plt.axis([0, len(b_z), -250, 250])
    plt.xlabel('r, px', fontsize=18)
    plt.ylabel('Conventional units', fontsize=18)

    plt.plot(xs, b_z)
    plt.plot(xs, b_phi)
    plt.plot(xs, j_z)
    plt.plot(xs, j_phi)
    plt.legend(['b_z', 'b_phi', 'j_z', 'j_phi'], loc='upper right')

    plt.savefig(os.path.join(os.path.dirname(filename_1), 'fields.png'))

    ### CSV GEN
    import csv
    with open(os.path.join(os.path.dirname(filename_1), 'fields.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(('b_z, G', 'b_phi, deg', 'j_z, G/px', 'j_phi, deg', 'r, px'))
        for i in range(len(b_z)):
            w.writerow([b_z[i], b_phi[i], j_z[i], j_phi[i], i])


    print(xs)



if __name__ == '__main__':
    filename_field = sys.argv[1]
    #f2 = sys.argv[2]
    #visualize_plot(filename_field, f2)
    #raise

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

    slice_values = np.array(slice_data.values, dtype=np.float64)[:, :3]
    slice_grid = np.array(slice_data.values, dtype=np.float64)[:, 3:6]

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

    val_cylindrical = np.concatenate([val_r, val_phi, val_n], axis=1)

    CUT_RADIUS = 14

    print(p_n)
    print(p_v)
    print(p_w)
    print(val_cylindrical.shape)

    from scipy.interpolate import LinearNDInterpolator

    regular_points = np.arange(-CUT_RADIUS, CUT_RADIUS + 1)
    V_NEW, W_NEW = np.meshgrid(regular_points, regular_points)
    interp_r = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 0])
    interp_phi = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 1])
    interp_n = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 2])

    V_NEW = V_NEW.astype(np.float64)
    W_NEW = W_NEW.astype(np.float64)

    R_VAL = interp_r(V_NEW, W_NEW)[..., None].astype(np.float64)
    PHI_VAL = interp_phi(V_NEW, W_NEW)[..., None].astype(np.float64)
    N_VAL = interp_n(V_NEW, W_NEW)[..., None].astype(np.float64)

    #print(V_NEW * N_VAL)

    N_VAL = N_VAL[..., 0]

    # V_C = np.sum(V_NEW * (N_VAL - np.min(N_VAL))) / np.sum((N_VAL - np.min(N_VAL)))
    # W_C = np.sum(W_NEW * (N_VAL - np.min(N_VAL))) / np.sum((N_VAL - np.min(N_VAL)))

    V_C, W_C = 2.7567683836490704, 0.5224803310849225  # CUT RADIUS 14

    print('center: ', V_C, W_C)

    # RE-INTERPOLATE AFTER FIRST ESTIMATION

    V_NEW = V_NEW.astype(np.float64) + V_C
    W_NEW = W_NEW.astype(np.float64) + W_C

    R_VAL = interp_r(V_NEW, W_NEW)[..., None].astype(np.float64)
    PHI_VAL = interp_phi(V_NEW, W_NEW)[..., None].astype(np.float64)
    N_VAL = interp_n(V_NEW, W_NEW)[..., None].astype(np.float64)

    result = np.concatenate([R_VAL, PHI_VAL, N_VAL], axis=-1)

    V_NEW = V_NEW - V_C
    W_NEW = W_NEW - W_C
    VW = np.concatenate([V_NEW[..., None], W_NEW[..., None]], axis=-1)

    print(VW.shape)

    VW_r = np.linalg.norm(VW, axis=-1, keepdims=False)
    # e_phi(tau) = (-sin_phi, cos_phi)
    # e_normal = (cos_phi, sin_phi)
    VW_cos_phi = V_NEW / VW_r
    VW_sin_phi = W_NEW / VW_r

    print(VW_cos_phi)
    print(VW_sin_phi)
    raise

    N_VAL = N_VAL[..., 0]
    PHI_VAL = PHI_VAL[..., 0]

    fn = []
    fphi = []
    for i in range(CUT_RADIUS):
        r_cur = np.where((VW_r >= i) & (VW_r < (i + 1)))
        n_cur = N_VAL[r_cur]
        phi_cur = PHI_VAL[r_cur]
        fn.append(np.mean(n_cur))
        fphi.append(np.mean(phi_cur))

    data = {'b_phi': np.array(fphi), 'b_z': np.array(fn)}
    #
    # print(fn)
    # print(fphi)
    #
    # bins = np.arange(CUT_RADIUS)
    # print(VW_r.shape)
    # print(N_VAL.shape)
    # val_phi = np.arctan2(val_w, val_v)

    np.save(os.path.join(os.path.dirname(filename_csv),
                         os.path.splitext(os.path.basename(filename_csv))[0] + '_data_29.npy'), data, allow_pickle=True)



    #print(centered_grid)
    #print(result.shape)
    raise


    inter_x = RegularGridInterpolator((x, y, z), cx)
    inter_y = RegularGridInterpolator((x, y, z), cy)
    inter_z = RegularGridInterpolator((x, y, z), cz)

    # from scipy.ndimage import gaussian_filter

    # smooth = gaussian_filter(cx, sigma=1)

    pts = np.array([[15.2, 16.2, 18.3], [3.3, 5.2, 7.1]])
    print(inter_x(pts))
    print(inter_y(pts))

    print(cx.shape)

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
