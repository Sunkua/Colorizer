import redis
import glob
import os
import h5py
import hdf5plugin

def getBaseNames(items):
    for i, item in enumerate(items):
        items[i] = os.path.basename(item)
    return items

r = redis.Redis(host='redis', port=6379, db=0)
test = r.llen("videos")



while(r.llen("videos") > 0):
    r.lpop("videos")

r.set("workerid",0)

videos = glob.glob('/network-ceph/pgrundmann/youtube_processed_small/*.mp4')
videos = getBaseNames(videos)

dataset = "/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5"
with h5py.File(dataset, 'r') as f_data:
    for video in videos:
        if not video in f_data:
            r.lpush("videos", video)