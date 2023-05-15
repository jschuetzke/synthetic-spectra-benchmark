#!/usr/bin/env python
# coding: utf-8

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
test_var = 0.04

n_test = 9 # predefined variations, just here for clarity
n_val = 10
n_train = 50

rng = np.random.default_rng(2022)

def vary_peaks(position_list, height_list):
    # since we set a boundary parameter, positions should never exceed range
    # still applying clip to be sure
    new_positions = np.clip(np.array([
        rng.integers(f-shift_range, f+shift_range) for f in position_list
        ]), 0, 4999)
    new_heights = np.clip(np.array([
        rng.uniform(f-variation_range, f+variation_range) for f in height_list
        ]), 0, 2)
    return new_positions, new_heights

def main():   
    with open('dataset_configs/dataset500.json', 'r') as file:
        config = json.load(file)
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
        scan[0,peak_positions] = peak_heights # exact reference
        # two variants with exact heights and -25/+25 position shifts
        if int(phase) in [8,318]: # exclude classes with overlapping positions
            pos_left = np.clip(np.array(peak_positions)-17, 0, 4999)
            pos_right = np.clip(np.array(peak_positions)+17, 0, 4999)
        else:
            pos_left = np.clip(np.array(peak_positions)-test_shift, 0, 4999)
            pos_right = np.clip(np.array(peak_positions)+test_shift, 0, 4999)
        scan[1,pos_left] = peak_heights
        scan[2,pos_right] = peak_heights
        # manual modifications to prevent overlap in test data
        # change lines for modified config!
        if int(phase) in [164,281,219,347]: # exclude classes with overlapping ints
            new_hi_hi = np.array([f+0.025 if f != 1. else f for f in peak_heights])
            new_hi_lo = np.array([f-0.025 if f != 1. else f for f in peak_heights])
        else:
            new_hi_hi = np.array([f+test_var if f != 1. else f for f in peak_heights])
            new_hi_lo = np.array([f-test_var if f != 1. else f for f in peak_heights])
        scan[3,peak_positions] = np.clip(new_hi_hi,0,2)
        scan[4,pos_left] = np.clip(new_hi_hi,0,2)
        scan[5,pos_right] = np.clip(new_hi_hi,0,2)
        scan[6,peak_positions] = np.clip(new_hi_lo,0,2)
        scan[7,pos_left] = np.clip(new_hi_lo,0,2)
        scan[8,pos_right] = np.clip(new_hi_lo,0,2)
        for j in range(n_test):
            x_test[i*n_test+j] = gaussian_filter1d(scan[j], 2, mode='constant')
        y_test[i*n_test:(i+1)*n_test] = i   

    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    main()
