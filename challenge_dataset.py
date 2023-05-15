#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# GLOBAL PARAMETERS

shift_range = 50 # for now, we shift all peaks independently
variation_range = 0.05 # +/- of absolute height for each peak
kernel_range = (2, 5) # min and max for guassian kernel sizes

# less variation for test data
test_shift = 30 
test_var = 0.03

n_test = 6 # predefined variations, just here for clarity
n_val = 10
n_train = 50

rng = np.random.default_rng(2022)

# PARAMETERS
n_datapoints = 5000
boundary = 100
n_classes = 9
min_peaks = 2
max_peaks = 10
max_height = 100

def generate_config():
    config = {
        'datapoints' : n_datapoints,
        'boundary' : boundary,
        'classes' : n_classes,
        'min_peaks' : min_peaks,
        'max_peaks' : max_peaks,
        'max_height' : max_height
    }
    spectra = {
        # 9 classes for minor peak detection
        # triplet 1
        0 : {"peak_positions" : [700, 3000], "peak_heights" : [0.7, 1.0]},
        1 : {"peak_positions" : [700, 1800, 3000], "peak_heights" : [0.7, 0.045, 1.0]},
        2 : {"peak_positions" : [700, 3000, 4200], "peak_heights" : [0.7, 1.0, 0.045]},
        # triplet 2
        3 : {"peak_positions" : [920, 1115], "peak_heights" : [1.0, 0.4]},
        4 : {"peak_positions" : [923, 1112, 1500], "peak_heights" : [1.0, 0.4, 0.045]},
        5 : {"peak_positions" : [918, 1116, 1630], "peak_heights" : [1.0, 0.4, 0.045]},
        # triplet 3
        6 : {"peak_positions" : [500, 2100], "peak_heights" : [0.9, 1.0]},
        7 : {"peak_positions" : [500, 2100, 2260], "peak_heights" : [0.9, 1.0, 0.045]},
        8 : {"peak_positions" : [500, 2100, 2260, 2750], "peak_heights" : [0.9, 1.0, 0.045, 0.045]},

        # 9 classes for position overlap
        # triplet 4
        9 : {"peak_positions" : [700, 3000], "peak_heights" : [1.0, 0.7]},
        10 : {"peak_positions" : [700, 3070], "peak_heights" : [1.0, 0.7]},
        11 : {"peak_positions" : [700, 3140], "peak_heights" : [1.0, 0.7]},
        # triplet 5
        12 : {"peak_positions" : [1230, 2103], "peak_heights" : [0.15, 1.0]},
        13 : {"peak_positions" : [1300, 2101], "peak_heights" : [0.15, 1.0]},
        14 : {"peak_positions" : [1370, 2098], "peak_heights" : [0.15, 1.0]},
        # triplet 6
        15 : {"peak_positions" : [1230, 1750, 4700], "peak_heights" : [0.15, 0.2, 1.0]},
        16 : {"peak_positions" : [1300, 1820, 4700], "peak_heights" : [0.15, 0.2, 1.0]},
        17 : {"peak_positions" : [1370, 1890, 4700], "peak_heights" : [0.15, 0.2, 1.0]},

        # 9 classes for height overlap
        # triplet 7
        18 : {"peak_positions" : [920, 1500], "peak_heights" : [0.08, 1.0]},
        19 : {"peak_positions" : [920, 1500], "peak_heights" : [0.155, 1.0]},
        20 : {"peak_positions" : [920, 1500], "peak_heights" : [0.23, 1.0]},
        # triplet 8
        21 : {"peak_positions" : [1147, 2261], "peak_heights" : [1.0, 0.20]},
        22 : {"peak_positions" : [1151, 2258], "peak_heights" : [1.0, 0.275]},
        23 : {"peak_positions" : [1153, 2262], "peak_heights" : [1.0, 0.35]},
        # triplet 9
        24 : {"peak_positions" : [302, 2750, 4198], "peak_heights" : [0.08, 1.0, 0.22]},
        25 : {"peak_positions" : [300, 2750, 4200], "peak_heights" : [0.155, 1.0, 0.21]},
        26 : {"peak_positions" : [299, 2750, 4202], "peak_heights" : [0.23, 1.0, 0.23]},
    }

    config['spectra'] = spectra
    with open('dataset_configs/challenge.json', 'w') as file:
        json.dump(config, file)
    return config

def vary_peaks(position_list, height_list):
    # since we set a boundary parameter, positions should never exceed range
    # still applying clip to be sure
    new_positions = np.clip(np.array([
        rng.integers(f-shift_range, f+shift_range) for f in position_list
        ]), 0, 4999)
    new_heights = np.clip(np.array([
        rng.uniform(f-variation_range, f+variation_range) if f != 1. else f for f in height_list
        ]), 0, 2)
    return new_positions, new_heights

def main():   
    # with open('dataset_configs/challenge.json', 'r') as file:
    #     config = json.load(file)
    config = generate_config()
    if not os.path.exists('challenge_dataset'):
        os.makedirs('challenge_dataset')
    datapoints = config['datapoints']
    n_classes = config['classes']
    # initialize arrays to fill
    x_train = np.zeros([(n_train*n_classes), datapoints])
    x_val = np.zeros([(n_val*n_classes), datapoints])
    x_test = np.zeros([(n_test*n_classes), datapoints])
    y_train = np.zeros(n_train*n_classes)
    y_val = np.zeros(n_val*n_classes)
    y_test = np.zeros(n_test*n_classes)

    spectra = config['spectra']
    for i, phase in enumerate(tqdm(spectra.keys())):
        peak_positions = np.array(spectra[phase]['peak_positions'])
        peak_heights = np.array(spectra[phase]['peak_heights'])
        
        # train data
        for j in range(n_train):
            scan = np.zeros(datapoints)
            # apply shifts
            new_pos, new_hi = vary_peaks(peak_positions, peak_heights)
            scan[new_pos] = new_hi
            # convolve with gaussian kernel
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                     mode='constant')
            x_train[(i*n_train)+j] = scan
        y_train[i*n_train:(i+1)*n_train] = i
        
        # validation data
        for j in range(n_val):
            scan = np.zeros(datapoints)
            # apply shifts
            new_pos, new_hi = vary_peaks(peak_positions, peak_heights)
            scan[new_pos] = new_hi
            # convolve with gaussian kernel
            scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                     mode='constant')
            x_val[(i*n_val)+j] = scan
        y_val[i*n_val:(i+1)*n_val] = i
        
        # test - use exact values + defined variations
        scan = np.zeros([n_test, datapoints])
        # two variants with exact heights and -25/+25 position shifts
        pos_left = np.clip(np.array(peak_positions)-test_shift, 0, 4999)
        pos_right = np.clip(np.array(peak_positions)+test_shift, 0, 4999)
        new_hi_hi = np.array([f+test_var if f != 1. else f for f in peak_heights])
        new_hi_lo = np.array([f-test_var if f != 1. else f for f in peak_heights])
        if int(phase) in range(9):
            scan[0,peak_positions] = peak_heights
            scan[1,pos_left] = peak_heights
            scan[2,pos_right] = peak_heights
            scan[3,peak_positions] = np.clip(new_hi_lo,0,2)
            scan[4,pos_left] = np.clip(new_hi_lo,0,2)
            scan[5,pos_right] = np.clip(new_hi_lo,0,2)
        if int(phase) in [9,12,15]:
            scan[0,peak_positions] = peak_heights
            scan[1,pos_right] = peak_heights
            scan[2,peak_positions] = np.clip(new_hi_hi,0,2)
            scan[3,pos_right] = np.clip(new_hi_hi,0,2)
            scan[4,peak_positions] = np.clip(new_hi_lo,0,2)
            scan[5,pos_right] = np.clip(new_hi_lo,0,2)
        if int(phase) in [10,13,16]:
            scan[0,pos_left] = peak_heights
            scan[1,pos_right] = peak_heights
            scan[2,pos_left] = np.clip(new_hi_hi,0,2)
            scan[3,pos_right] = np.clip(new_hi_hi,0,2)
            scan[4,pos_left] = np.clip(new_hi_lo,0,2)
            scan[5,pos_right] = np.clip(new_hi_lo,0,2)
        if int(phase) in [11,14,17]:
            scan[0,peak_positions] = peak_heights
            scan[1,pos_left] = peak_heights
            scan[2,peak_positions] = np.clip(new_hi_hi,0,2)
            scan[3,pos_left] = np.clip(new_hi_hi,0,2)
            scan[4,peak_positions] = np.clip(new_hi_lo,0,2)
            scan[5,pos_left] = np.clip(new_hi_lo,0,2)
        if int(phase) in [18,21,24]:
            scan[0,peak_positions] = peak_heights
            scan[1,pos_left] = peak_heights
            scan[2,pos_right] = peak_heights
            scan[3,peak_positions] = np.clip(new_hi_hi,0,2)
            scan[4,pos_left] = np.clip(new_hi_hi,0,2)
            scan[5,pos_right] = np.clip(new_hi_hi,0,2)
        if int(phase) in [19,22,25]:
            scan[0,peak_positions] = np.clip(new_hi_lo,0,2)
            scan[1,pos_left] = np.clip(new_hi_lo,0,2)
            scan[2,pos_right] = np.clip(new_hi_lo,0,2)
            scan[3,peak_positions] = np.clip(new_hi_hi,0,2)
            scan[4,pos_left] = np.clip(new_hi_hi,0,2)
            scan[5,pos_right] = np.clip(new_hi_hi,0,2)
        if int(phase) in [20,23,26]:
            scan[0,peak_positions] = peak_heights
            scan[1,pos_left] = peak_heights
            scan[2,pos_right] = peak_heights
            scan[3,peak_positions] = np.clip(new_hi_lo,0,2)
            scan[4,pos_left] = np.clip(new_hi_lo,0,2)
            scan[5,pos_right] = np.clip(new_hi_lo,0,2)
        for j in range(n_test):
            x_test[i*n_test+j] = gaussian_filter1d(scan[j], 2, mode='constant')
        y_test[i*n_test:(i+1)*n_test] = i   

    np.save('challenge_dataset/x_train.npy', x_train)
    np.save('challenge_dataset/y_train.npy', y_train)
    np.save('challenge_dataset/x_val.npy', x_val)
    np.save('challenge_dataset/y_val.npy', y_val)
    np.save('challenge_dataset/x_test.npy', x_test)
    np.save('challenge_dataset/y_test.npy', y_test)

if __name__ == '__main__':
    main()
