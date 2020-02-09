import random
from skvideo import io
import numpy as np
import cv2
import skvideo
import logging
from tqdm import tqdm
import os
import redis

r = redis.Redis(host='redis', port=6379, db=0)

worker_id = r.incr("workerid", amount=1)
length = r.llen("videos")
done = 0
histogram = None
bins = np.arange(start=0,stop=1,step=(1.0/316))
while(r.llen("videos") > 0):
    video = (r.lpop("videos")).decode('utf-8')
    video = '/network-ceph/pgrundmann/youtube_processed_small/' + video
    video = skvideo.io.vread(video)
    video_float = np.zeros(video.shape, dtype=np.float32)

    for i, frame in enumerate(video):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        frame = frame.astype(np.float32)
        video_float[i] = frame

    video_float /= 255.0
    a = video_float[:,:,:,1]
    b = video_float[:,:,:,2]
    a = a.flatten()
    b = b.flatten()
    if histogram is None:
        histogram,_,_= np.histogram2d(a,b,bins=(17,17), range=((0,1), (0,1)), density=True)
        histogram /= histogram.sum()
    else:
        h_, _,_ =np.histogram2d(a,b,bins=(17,17), range=((0,1), (0,1)), density=True)
        h_ /= h_.sum()
        histogram += h_
        histogram /= 2.0
    
    print(done, " videos finished")
    done += 1
np.save("/network-ceph/pgrundmann/maschinelles_sehen/histogram_mean/np_means/" + str(worker_id) + "_hist_a.np",histogram)
np.save("/network-ceph/pgrundmann/maschinelles_sehen/histogram_mean/np_means/" + str(worker_id) + "_hist_b.np",histogram)

        
