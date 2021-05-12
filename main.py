from magnetic.sav2vtk import *
import os
import numpy as np
import cv2
from glob import glob
import json
from magnetic.utils import save_targets_images, plot_data
from magnetic.mathops import curl
import scipy.io as scio
from magnetic.utils import get_times_from_folder, rename_files_accordingly_to_timecodes


if __name__ == '__main__':
    convert_folder('/media/sunshine/HDD/thz_events/20120704/NLFFFE_sav', 'jB_closure', func=box2directions2vtk, n_jobs=8)

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
    # regular_grid('/home/sunshine/data/regular_grid')
    #read_looptrace('C:\\AppsData\\NLFFFE\\BASEMAPS\\sav\\loops\\tracesAIA_94NORH_NLFFFE_170903_224642.dat')
