import torch
import random
from skimage import color
from skvideo import io
import numpy as np
import cv2
import skvideo
from torch.multiprocessing import Queue
import logging
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import h5py
import hdf5plugin




class MultiProcDataset(torch.utils.data.IterableDataset):
    # videos is a multiprocessing-queue
    def __init__(self, videos, sequence_length=16):
        self.videos = videos
        self.sequence_length = sequence_length
        
        self.active_video = None
        self.l_dataset = None
        self.resnet_dataset = None
        self.hist_dataset = None
        self.position = 0


    def __refresh_samples__(self):
        if self.active_video == None:
            if len(self.videos) == 0:
                raise StopIteration
            self.active_video = self.videos.pop()
            self.position = 0
            self.l_dataset = self.h5_file[self.active_video + "/images_l"]
            self.hist_dataset = self.h5_file[self.active_video +"/images_hists"]
            self.resnet_dataset = self.h5_file[self.active_video +"/images_resnet"]

        self.first_iteration = torch.zeros(1, dtype=torch.bool)
        if self.position == 0:
            self.first_iteration = torch.ones(1, dtype=torch.bool)
        
        pos = self.position
        batchsize = self.sequence_length
        self.l_sample = torch.tensor(self.l_dataset[pos:pos+batchsize]).squeeze()
        self.l_sample = self.l_sample.unsqueeze(dim=0)
        self.hist_sample = torch.tensor(self.hist_dataset[pos:pos+batchsize]).unsqueeze(dim=0)
        self.resnet_sample = torch.tensor(self.resnet_dataset[pos:pos+batchsize]).unsqueeze(dim=0)
        self.resnet_sample = self.resnet_sample.permute(0,1,4,2,3)
        self.resnet_sample =  self.resnet_sample.reshape((self.resnet_sample.shape[0] * self.resnet_sample.shape[1], self.resnet_sample.shape[2], self.resnet_sample.shape[3], self.resnet_sample.shape[4]))
        if self.l_sample.shape[1] < batchsize:
            self.active_video = None
            raise StopIteration

        self.l_sample.pin_memory()
        self.resnet_sample.pin_memory()
        self.hist_sample.pin_memory()




    
        
    def __next__(self):
        self.__refresh_samples__()
        batch = [self.l_sample, self.first_iteration, self.resnet_sample, self.hist_sample]
        return batch
      
    
    def __iter__(self):
        self.h5_file = h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r')

        random.shuffle(self.videos)
        
           
        return self
  

