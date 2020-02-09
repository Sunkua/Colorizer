import glob
import numpy as np
import h5py
import hdf5plugin
from tqdm import tqdm
hdf5_files = glob.glob('/network-ceph/pgrundmann/video_dataset/video_dataset*.hdf5')


with h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'w') as f:
    files = []
    for dataset in tqdm(hdf5_files):
        with h5py.File(dataset, 'r') as f_data:
            for name in tqdm(f_data):
                if not name in f:
                    f_data.copy(name, f) 
