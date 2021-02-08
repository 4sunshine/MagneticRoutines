from magnetic.sav2vtk import *
import os
import numpy as np
import cv2
from glob import glob
import json
from magnetic.utils import put_text


if __name__ == '__main__':
    with open('/home/sunshine/data/2017_09_04/times.json', 'r') as f:
        times = json.load(f)
    times = times['times']
    target_dir = '/media/sunshine/HDD/Loops/concat'
    os.makedirs(target_dir, exist_ok=True)
    savs_94 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_94/*.sav'))
    savs_131 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_131/*.sav'))
    currents = sorted(glob('/media/sunshine/HDD/AppsData/NLFFFE/BASEMAPS/sav/Vert_*.sav'))
    imgs_94 = []
    imgs_131 = []
    currents_pos = []
    currents_neg = []
    for im_94, im_131, cur in zip(savs_94, savs_131, currents):
        imgs_94.append(np.flip(get_image2_from_sav(im_94), axis=0))
        imgs_131.append(np.flip(get_image2_from_sav(im_131), axis=0))
        current = np.flip(get_image2_from_sav(im_94), axis=0)
        current_pos = np.clip(current, 0, np.max(current))
        current_neg = -np.clip(current, np.min(current), 0)
        currents_pos.append(current_pos)
        currents_neg.append(current_neg)
    imgs_94 = clip_neg_and_max_divide(imgs_94)
    imgs_131 = clip_neg_and_max_divide(imgs_131)
    currents_pos = clip_neg_and_max_divide(currents_pos)
    currents_neg = clip_neg_and_max_divide(currents_neg)
    for i, (im_94, im_131, t, c_pos, c_neg) in enumerate(zip(imgs_94, imgs_131, times, currents_pos, currents_neg)):
        result = np.zeros((*np.shape(im_94), 3), dtype=np.uint8)
        result[..., 1] = im_94
        result[..., 2] = im_131

        current_base = np.ones((*np.shape(im_94), 3), dtype=np.uint8)
        current_base_col = np.array([255, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        current_base = current_base * current_base_col
        current_base = cv2.cvtColor(current_base, cv2.COLOR_BGR2HSV)
        current_base[..., -1] = c_pos
        current_base = cv2.cvtColor(current_base, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(target_dir, f'KEK_{i:03d}.png'), current_base)

        result = cv2.resize(result, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        result = put_text(result, 'AIA_094', color=(0, 255, 0), origin=(0.4, 0.95))
        result = put_text(result, 'AIA_131', color=(0, 0, 255), origin=(0.6, 0.95))
        result = put_text(result, t, origin=(0, 0.95))
        cv2.imwrite(os.path.join(target_dir, f'{i:03d}.png'), result)

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
    # regular_grid('/home/sunshine/data/regular_grid')
    #read_looptrace('C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat')
