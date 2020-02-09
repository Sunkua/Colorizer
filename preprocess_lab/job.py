import h5py
import hdf5plugin
import skvideo
from skvideo import io
import numpy as np
import cv2
from tqdm import tqdm
import redis


def loadVideos(path):
    # load all .mp4-files from path into list
    train_list = path + "/train_filenames.txt"
    test_list  =path + "/test_filenames.txt"

    f = open(train_list, "r")
    training_names = f.readlines()
    training_names = [x.strip() for x in training_names]
    f.close()

    f = open(test_list, "r")
    test_names = f.readlines()
    test_names = [x.strip() for x in test_names]
    f.close()
    return training_names, test_names

#training, _ = loadVideos("/network-ceph/pgrundmann/youtube_processed_small")
r = redis.Redis(host='redis', port=6379, db=0)
worker_id = r.incr("workerid", amount=1)
with h5py.File('/network-ceph/pgrundmann/video_dataset/video_dataset_' + str(worker_id) + '.hdf5', 'w') as f:
    while(r.llen("videos") > 0):
        video_filename = (r.lpop("videos")).decode('utf-8')
        fname = '/network-ceph/pgrundmann/youtube_processed_small/' + video_filename
        video = skvideo.io.vread(fname)
        video = np.moveaxis(video, 3, 1)
        video_processed = np.empty(video.shape, dtype=np.float32)
        for i, frame in enumerate(video):
            frame = np.moveaxis(frame, 0,2)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
            frame = frame.astype(np.float32)
            frame /= 255.0    
            frame = np.moveaxis(frame, 2,0)
            video_processed[i] = frame
        data_grp = f.create_group(video_filename)
        video_ds = data_grp.create_dataset('video', video_processed.shape, dtype=np.float32, data=video_processed,**hdf5plugin.Blosc(cname='blosclz', clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))

