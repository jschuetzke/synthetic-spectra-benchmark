#!/usr/bin/env python
# coding: utf-8

"generate a dataset config (json) based on predefined parameters"

import json
import numpy as np

# PARAMETERS
n_datapoints = 5000
boundary = 100
n_classes = 500
min_peaks = 2
max_peaks = 10
max_height = 100
distribution = 'gamma' # 'uniform' alternative

def main():
    rng = np.random.default_rng(2022)
    config = {
        'datapoints' : n_datapoints,
        'boundary' : boundary,
        'classes' : n_classes,
        'min_peaks' : min_peaks,
        'max_peaks' : max_peaks,
        'max_height' : max_height
        }
    spectra = {}
    for phase in range(n_classes):
        if distribution == 'gamma':
            n_peaks = np.round((rng.gamma(1.2,1.2)+2)).astype(int) # favor less peaks
        elif distribution == 'uniform':
            n_peaks = rng.integers(min_peaks, max_peaks, endpoint=True)
        else:
            raise ValueError(f'unknown distribution {distribution}')
        peak_positions = rng.integers(boundary, n_datapoints-boundary, 
                                      n_peaks)[:max_peaks]
        peak_heights = rng.integers(1, max_height, n_peaks, endpoint=True)[:max_peaks]
        # scale peak heights according to highest peak in the list
        # sets highest peak in list to 1 and scales others accordingly
        peak_heights = np.round(peak_heights / np.max(peak_heights), 3)
        phase_dict = {'peak_positions': np.sort(peak_positions).tolist(),
                      'peak_heights' : peak_heights[np.argsort(peak_positions)].tolist()}
        spectra[phase] = phase_dict
    
    config['spectra'] = spectra
    with open('dataset_configs/dataset500.json', 'w') as file:
        json.dump(config, file)

if __name__ == '__main__':
    main()
