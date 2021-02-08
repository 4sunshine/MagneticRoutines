import cv2
import numpy as np
from magnetic.sav2vtk import *
import json
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import spines
from tqdm import tqdm


def plot_data(data, save_path):
    """Data should have the shape: [Len_data, N_lines]"""
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    channel_colors = ['#00ff00', '#0000ff', '#000000']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    ax.set_xlabel('frame #')
    ax.set_ylabel('I, A')
    ax.set_facecolor('white')
    for child in ax.get_children():
        if isinstance(child, spines.Spine):
            child.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    fig.patch.set_facecolor('white')
    plt.xlim(0, 37)
    #plt.ylim(100, 255)
    #data_len, n_lines = np.shape(data)[:2]
    for i in range(len(data)):
        plt.plot(data[i], color=channel_colors[i])
        #plt.plot(data_len - 1, data[-1, i], color=channel_colors[i], marker='o')
    plt.savefig(save_path)
    plt.close()


def put_text(image, text, color=(255, 255, 255), origin=(0, 1), thickness=2):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    h, w = np.shape(image)[:2]
    org = (int(origin[0] * w), int(origin[1] * h))
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    # Using cv2.putText() method
    return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


def save_images():
    with open('/home/sunshine/data/2017_09_04/times.json', 'r') as f:
        times = json.load(f)
    times = times['times']
    target_dir = '/media/sunshine/HDD/Loops/concat'
    os.makedirs(target_dir, exist_ok=True)
    savs_94 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_94/*.sav'))[:]
    savs_131 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_131/*.sav'))[:]
    currents = sorted(glob('/media/sunshine/HDD/AppsData/NLFFFE/npy_field/Bnlfffe*.npy'))[:]
    imgs_94 = []
    imgs_131 = []
    currents_pos = []
    currents_neg = []
    alphas = []

    for im_94, im_131, cur in zip(savs_94, savs_131, currents):
        imgs_94.append(np.flip(get_image2_from_sav(im_94), axis=0))
        imgs_131.append(np.flip(get_image2_from_sav(im_131), axis=0))
        _, _, cz = get_curl_from_np_box(cur)
        current = np.transpose(cz[..., 0])  # DUE TO Y, X ORDER ON IMAGE
        #current = np.flip(get_image2_from_sav(cur), axis=0)
        current_pos = np.clip(current, 0, np.max(current))
        alpha_pos = current_pos.astype(np.int32)
        alpha_pos = np.where(alpha_pos > 0, 1, 0)
        alphas.append(alpha_pos)
        current_neg = -np.clip(current, np.min(current), 0)
        currents_pos.append(current_pos)
        currents_neg.append(current_neg)
    imgs_94 = clip_neg_and_max_divide(imgs_94)
    imgs_131 = clip_neg_and_max_divide(imgs_131)
    currents_pos = clip_neg_and_max_divide(currents_pos)
    currents_neg = clip_neg_and_max_divide(currents_neg)
    alphas = np.array(alphas)
    for i, (im_94, im_131, t, c_pos, c_neg, a) in enumerate(zip(imgs_94, imgs_131, times,
                                                                currents_pos, currents_neg, alphas)):
        result = np.zeros((*np.shape(im_94), 3), dtype=np.uint8)
        result[..., 1] = im_94
        result[..., 2] = im_131

        current_base = np.ones((*np.shape(im_94), 3), dtype=np.uint8)
        current_base_col = np.array([255, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        current_base = current_base * current_base_col
        a = a[..., np.newaxis]
        current_base = current_base * a
        current_base = current_base.astype(np.uint8)

        current_base = cv2.cvtColor(current_base, cv2.COLOR_BGR2HSV)
        current_base[..., -1] = c_pos

        current_base_n = np.ones((*np.shape(im_94), 3), dtype=np.uint8)
        current_base_col = np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        current_base_n = current_base_n * current_base_col
        current_base_n = current_base_n * (1 - a)
        current_base_n = current_base_n.astype(np.uint8)
        current_base_n = cv2.cvtColor(current_base_n, cv2.COLOR_BGR2HSV)
        current_base_n[..., -1] = c_neg

        current_base = current_base + current_base_n

        current_base = cv2.cvtColor(current_base, cv2.COLOR_HSV2BGR)
        c_copy = cv2.resize(current_base, (800, 800), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(target_dir, f'CURRENT_{i:03d}.png'), c_copy)

        c_copy = put_text(c_copy, t, origin=(0, 0.95))
        c_copy = put_text(c_copy, 'CurlB_z-', color=(255, 0, 0), origin=(0.4, 0.95))
        c_copy = put_text(c_copy, 'CurlB_z+', color=(255, 0, 255), origin=(0.6, 0.95))
        cv2.imwrite(os.path.join(target_dir, f'CURRENT_{i:03d}.png'), c_copy)

        result = cv2.addWeighted(current_base, 0.3, result, 0.7, 0)

        result = cv2.resize(result, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        result = put_text(result, 'AIA_094', color=(0, 255, 0), origin=(0.4, 0.95))
        result = put_text(result, 'AIA_131', color=(0, 0, 255), origin=(0.6, 0.95))
        result = put_text(result, 'CurlB_z', color=(255, 255, 255), origin=(0.8, 0.95))
        result = put_text(result, t, origin=(0, 0.95))
        cv2.imwrite(os.path.join(target_dir, f'{i:03d}.png'), result)


def save_targets_images():
    RADIUS = 5
    BRIGHTEST_YELLOWS = ((0, 255, 255), (0, 124, 255), (255, 158, 0))
    LINE = 94

    target_dir = f'/media/sunshine/HDD/Loops/target_loops_circles_{LINE}'
    os.makedirs(target_dir, exist_ok=True)
    savs_94 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_94/*.sav'))[:]
    savs_131 = sorted(glob('/media/sunshine/HDD/Loops/loops_final_131/*.sav'))[:]
    currents = sorted(glob('/media/sunshine/HDD/AppsData/NLFFFE/npy_field/Bnlfffe*.npy'))[:]
    imgs_94 = []
    imgs_131 = []
    currents_pos = []
    currents_neg = []
    alphas = []
    total_currents = []

    for im_94, im_131, cur in zip(savs_94, savs_131, currents):
        imgs_94.append(np.flip(get_image2_from_sav(im_94), axis=0))
        imgs_131.append(np.flip(get_image2_from_sav(im_131), axis=0))
        _, _, cz = get_curl_from_np_box(cur)
        current = np.transpose(cz[..., 0])  # DUE TO Y, X ORDER ON IMAGE
        #current = np.flip(get_image2_from_sav(cur), axis=0)
        total_currents.append(current)
        current_pos = np.clip(current, 0, np.max(current))
        alpha_pos = current_pos.astype(np.int32)
        alpha_pos = np.where(alpha_pos > 0, 1, 0)
        alphas.append(alpha_pos)
        current_neg = -np.clip(current, np.min(current), 0)
        currents_pos.append(current_pos)
        currents_neg.append(current_neg)
    imgs_94 = clip_neg_and_max_divide(imgs_94)
    imgs_131 = clip_neg_and_max_divide(imgs_131)
    _, h, w = np.shape(imgs_94)

    total_currents = np.array(total_currents)

    with open('/home/sunshine/data/2017_09_04/times.json', 'r') as f:
        times = json.load(f)
    times = times['times']

    looptrace = f'/media/sunshine/HDD/Loops/loops_refactor_{LINE}/traces_AIA_{LINE}NORH_NLFFFE_170904_055842.dat'
    loops, ends = read_looptrace(looptrace)
    signals = []
    mean_signal = {}
    for k, v in loops.items():
        signal = np.array(v['signal'])
        #signals.append(np.mean(signal))
        mean_signal[k] = np.mean(signal)

    mean_signal = dict(sorted(mean_signal.items(), key=lambda item: item[1], reverse=True))
    # TOP 3 BRIGHTEST LOOPS
    target_loops = []
    for i, k in enumerate(mean_signal.keys()):
        coords = np.round(np.array(loops[k]['points'])).astype(np.int32)
        # due to image different coordinate systems
        coords[:, 1] = h - coords[:, 1]
        target_loops.append(coords)
        if i == 2:
            break

    #currents_pos = clip_neg_and_max_divide(currents_pos)
    #currents_neg = clip_neg_and_max_divide(currents_neg)
    #alphas = np.array(alphas)

    begin_currents = []
    end_currents = []

    for i, (im_94, im_131, t, current) in enumerate(zip(imgs_94, imgs_131, times, total_currents)):
        h, w = np.shape(im_94)[:2]
        result = np.zeros((*np.shape(im_94), 3), dtype=np.uint8)
        result[..., 1] = im_94
        result[..., 2] = im_131

        # current_base = np.ones((*np.shape(im_94), 3), dtype=np.uint8)
        # current_base_col = np.array([255, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        # current_base = current_base * current_base_col
        # a = a[..., np.newaxis]
        # current_base = current_base * a
        # current_base = current_base.astype(np.uint8)
        #
        # current_base = cv2.cvtColor(current_base, cv2.COLOR_BGR2HSV)
        # current_base[..., -1] = c_pos
        #
        # current_base_n = np.ones((*np.shape(im_94), 3), dtype=np.uint8)
        # current_base_col = np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        # current_base_n = current_base_n * current_base_col
        # current_base_n = current_base_n * (1 - a)
        # current_base_n = current_base_n.astype(np.uint8)
        # current_base_n = cv2.cvtColor(current_base_n, cv2.COLOR_BGR2HSV)
        # current_base_n[..., -1] = c_neg
        #
        # current_base = current_base + current_base_n
        #
        # current_base = cv2.cvtColor(current_base, cv2.COLOR_HSV2BGR)
        # c_copy = cv2.resize(current_base, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        #
        # cv2.imwrite(os.path.join(target_dir, f'CURRENT_{i:03d}.png'), c_copy)
        #
        # c_copy = put_text(c_copy, t, origin=(0, 0.95))
        # c_copy = put_text(c_copy, 'CurlB_z-', color=(255, 0, 0), origin=(0.4, 0.95))
        # c_copy = put_text(c_copy, 'CurlB_z+', color=(255, 0, 255), origin=(0.6, 0.95))
        # cv2.imwrite(os.path.join(target_dir, f'CURRENT_{i:03d}.png'), c_copy)
        #
        # result = cv2.addWeighted(current_base, 0.3, result, 0.7, 0)

        local_beg_current = []
        local_end_current = []

        for j, t_l in enumerate(target_loops):
            result = cv2.polylines(result, [t_l], False, BRIGHTEST_YELLOWS[j], 2)
            result = cv2.circle(result, (t_l[0, 0], t_l[0, 1]), RADIUS, BRIGHTEST_YELLOWS[j])
            result = put_text(result, f'{j}', color=BRIGHTEST_YELLOWS[j], origin=(t_l[0, 0] / w, t_l[0, 1] / h),
                              thickness=1)
            begin_pos = t_l[0]
            begin_current = np.sum(current[begin_pos[0] - RADIUS: begin_pos[0] + RADIUS,
                                   begin_pos[1] - RADIUS: begin_pos[1] + RADIUS])
            local_beg_current.append(begin_current)
            begin_pos = t_l[-1]
            end_current = np.sum(current[begin_pos[0] - RADIUS: begin_pos[0] + RADIUS,
                                 begin_pos[1] - RADIUS: begin_pos[1] + RADIUS])
            local_end_current.append(end_current)

        begin_currents.append(local_beg_current)
        end_currents.append(local_end_current)

        result = cv2.resize(result, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        result = put_text(result, 'AIA_094', color=(0, 255, 0), origin=(0.4, 0.95))
        result = put_text(result, 'AIA_131', color=(0, 0, 255), origin=(0.6, 0.95))
        # result = put_text(result, 'CurlB_z', color=(255, 255, 255), origin=(0.8, 0.95))
        result = put_text(result, t, origin=(0, 0.95))
        cv2.imwrite(os.path.join(target_dir, f'{i:03d}.png'), result)

    begin_currents = np.array(begin_currents)
    end_currents = np.array(end_currents)

    # with open(os.path.join(target_dir, f'begin_currents_{LINE}.npy'), 'wb') as f:
    np.save(os.path.join(target_dir, f'begin_currents_{LINE}.npy'), begin_currents)
    np.save(os.path.join(target_dir, f'end_currents_{LINE}.npy'), end_currents)
