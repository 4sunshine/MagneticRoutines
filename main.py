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
    print(data)
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


if __name__ == '__main__':
    filename_field = sys.argv[1]
    visualize_data(filename_field)
    raise
    filename_csv = sys.argv[2]
    curl, grid = box2curl2grid(filename_field)
    cx, cy, cz = curl
    x, y, z = grid

    from scipy.interpolate import RegularGridInterpolator
    import pandas as pd

    slice_data = pd.read_csv(filename_csv)

    slice_values = np.array(slice_data.values, dtype=np.float64)[:, 3:6]
    slice_grid = np.array(slice_data.values, dtype=np.float64)[:, :3]

    # PLANE R0, NORMAL
    central_point = np.array([190., 260., 15.], dtype=np.float64)
    plane_normal = np.array([-0.5069487516062562, 0.8619013500193842, 0.], dtype=np.float64)

    centered_grid = slice_grid - central_point[None, :]
    p_n, p_v, p_w = plane_vw(plane_normal)

    v_coords = centered_grid @ p_v[:, None]
    w_coords = centered_grid @ p_w[:, None]
    n_coords = centered_grid @ p_n[:, None]

    vw_coords = np.concatenate([v_coords, w_coords], axis=1)

    val_v = slice_values @ p_v[:, None]
    val_w = slice_values @ p_w[:, None]
    val_n = slice_values @ p_n[:, None]

    vw_val = np.concatenate([val_v, val_w], axis=1)

    val_r = np.linalg.norm(vw_val, axis=-1, keepdims=True)
    val_phi = np.arctan2(val_w, val_v)

    val_cylindrical = np.concatenate([val_r, val_phi, val_n], axis=1)

    CUT_RADIUS = 15

    print(p_n)
    print(p_v)
    print(p_w)
    print(val_cylindrical.shape)

    from scipy.interpolate import LinearNDInterpolator

    regular_points = np.arange(-CUT_RADIUS, CUT_RADIUS + 0.125, 0.125)
    V_NEW, W_NEW = np.meshgrid(regular_points, regular_points)
    interp_r = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 0])
    interp_phi = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 1])
    interp_n = LinearNDInterpolator(list(zip(vw_coords[:, 0], vw_coords[:, 1])), val_cylindrical[:, 2])

    R_VAL = interp_r(V_NEW, W_NEW)[..., None]
    PHI_VAL = interp_phi(V_NEW, W_NEW)[..., None]
    N_VAL = interp_n(V_NEW, W_NEW)[..., None]
    result = np.concatenate([R_VAL, PHI_VAL, N_VAL], axis=-1)

    np.save(os.path.join(os.path.dirname(filename_csv),
                         os.path.splitext(os.path.basename(filename_csv))[0] + '_planar_s125.npy'), result)
    print(result.shape)


    #print(centered_grid)
    print(slice_values.shape)
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
