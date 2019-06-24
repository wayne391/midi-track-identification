import os
import time
import argparse
import datetime
import numpy as np

import sys
sys.path.append('../')

from track_identifier.utils import features


def proc(tracks_dir, output_dir):
    start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # list files
    file_list = []
    for root, _, files in os.walk(tracks_dir):
        for file in files:
            if file.endswith(".npz"):
                file_list.append(os.path.join(root, file))
                    
    # print to check
    # for idx, file in enumerate(file_list):
    #    print(idx, file) 

    num_file = len(file_list)
    for fidx in range(num_file):
        filename = file_list[fidx]
        print('[{}/{}] {}'.format(fidx, num_file, filename), end='\r')
        
        entry = np.load(filename)
        track = entry['x']
        label = entry['y']
        
        # feature extraction
        X = features.extract_features(track.item().pianoroll)
        
        # save
        label = label.item()
        filename_res = os.path.splitext(os.path.basename(filename))[0]
        filename_res = os.path.join(output_dir, filename_res)
        np.savez(filename_res, x=X, y=label)

    # finish
    end_time = time.time()
    runtime = end_time - start_time
    print('\n\n[*] Finished!')
    print('> Elapsed Time:', str(datetime.timedelta(seconds=runtime))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks_dir', default='../data/tracks', help='tracks')
    parser.add_argument('--output_dir', default='../data/features', help='destination')
    args = parser.parse_args()
    proc(args.tracks_dir, args.output_dir)
