import torch
import h5py
import hdf5plugin
import numpy as np

class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, name, batchsize):
        self.name = name
        self.h5_file = h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r')
        self.__calc_len__(batchsize)
        self.datasets = [x for x in self.h5_file]
        self.pos = 0

    def __getitem__(self, idx):
        l_dataset = f[activeVideo+"/images_l"][idx]
        hist_dataset = f[activeVideo +"/images_hists"][idx]
        resnet_dataset = f[activeVideo+"/images_resnet"][idx]
        
        if idx == 0:
            
        #pass

    def __len__(self):
        return self.data_length

    def __calc_len__(self, batchsize):

        sample_count = self.h5_file[self.name+"/images_l"].shape[0]
        batches = sample_count // batchsize
        self.data_length = batches * batchsize



