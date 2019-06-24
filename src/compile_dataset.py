import os
import time
import argparse
import datetime
import numpy as np
from pypianoroll import Multitrack


def proc(midi_dir, output_dir):
    start_time = time.time()

    # list files
    file_list = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                file_list .append(os.path.join(root, file))
                    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # print to check
    # for idx, file in enumerate(file_list):
    #   print(idx, file) 

    # labeling
    num_file = len(file_list)
    track_cnt = 0
    for fidx in range(num_file):
        filename = file_list[fidx]
        multi = Multitrack(filename)

        print('[{}/{}] {}'.format(fidx, num_file, filename), end='\r')
       
        for track in multi.tracks: 
            
            tmp = track.name.split(' ')
            name = ' '.join(tmp[:-1]).strip()
            if name == 'Melody':
                label = 0
            elif name == 'Drums':
                label = 1
            elif name == 'Bass':
                label = 2
            elif any([name == 'Guitar', name == 'Piano',  name == 'Strings']):
                label = 3
            else:
                continue
            track_cnt += 1
            filename_out = os.path.splitext(os.path.basename(filename))[0] 
            filename_out = os.path.join(output_dir, filename_out+'_'+name)
            np.savez(filename_out, x=track, y=label)

    # finish
    end_time = time.time()
    runtime = end_time - start_time
    print('\n\n[*] Finished!')
    print('> Total: %d tracks', track_cnt)
    print('> Elapsed Time:', str(datetime.timedelta(seconds=runtime))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_dir', default='../data/raw/jazz_realbook', help='raw data')
    parser.add_argument('--output_dir', default='../data/tracks', help='destination')
    args = parser.parse_args()
    proc(args.midi_dir, args.output_dir)
