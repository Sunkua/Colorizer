import torch
import random
from skimage import color
import numpy as np
import torchvision
import cv2
from itertools import chain, cycle

class IterableVideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, videos, batchsize):
        self.videos = videos
        self.batchsize = batchsize
    
    @property
    def shuffled_data_list(self):
        return random.sample(self.videos, len(self.videos))
        
    def process_data(self, data):
        for x in data:
            worker= torch.utils.data.get_worker_info()
            worker_id = id(self) if worker is not None else -1
            
            yield x, worker_id
    
    def __iter__(self):
        return self.get_streams()

    def get_stream(self, videos):
        return chain.from_iterable(map(self.process_data, cycle(self.videos)))
   
    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list)
        for _ in range(self.batchsize)])
    
    @classmethod
    def split_datasets(cls, data_list, batchsize, max_workers):
        for n in range(max_workers,0,-1):
            if batchsize % n == 0:
                num_workers = n
                break
        
        split_size = batchsize // num_workers
        return [cls(data_list, batchsize=split_size)
                for _ in range(num_workers)]
        
class MultiStreamDataloader:

    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=None)
        for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))