import torch
import numpy as np
import torchvision
import cv2
import torchvision.models as models
import skvideo
from skvideo import io
import h5py
import hdf5plugin

class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, video, batchsize):
        #data = skvideo.io.ffprobe(video)['video']
        #self.length = np.int(data['@nb_frames'])
        self.video = video
        with h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r') as f:
            self.length = f[self.video + "/video"].shape[0] // batchsize
        self.batchsize = batchsize


        

    def __getitem__(self, idx):
        if idx == 0:
            self.data_file = h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r')
            self.data = self.data_file[self.video + "/video"]
        frame = self.data[idx*self.batchsize:(idx+1)*self.batchsize]
        frame = torch.tensor(frame, dtype=torch.float32)   
        return frame

        

    def __len__(self):
        return self.length

