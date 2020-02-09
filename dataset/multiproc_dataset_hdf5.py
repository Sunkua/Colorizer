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

# to_process: global video_filename_list
# processed: frames for the current process, processed exists for every process (cnt(batchsize))
def do_job(id, to_process, processed, batchsize):
    logger = torch.multiprocessing.log_to_stderr()
    with h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r') as f:
        video_finished = True
        firstIteration = True
        activeVideo = None
        video_dataset = None
        pos = 0
        l_dataset = None
        hist_dataset = None
        resnet_dataset = None
        while not to_process.empty():
            if video_finished:
                # try to get next one
                try:
                    logger.info(str(to_process.qsize()) + " items remaining in queue")
                    activeVideo = to_process.get(10)
                    l_dataset = f[activeVideo+"/images_l"][:]
                    hist_dataset = f[activeVideo +"/images_hists"][:]
                    resnet_dataset = f[activeVideo+"/images_resnet"][:]

                    l_dataset = torch.tensor(l_dataset, dtype=torch.float32)
                    hist_dataset = torch.tensor(hist_dataset, dtype=torch.float32)
                    resnet_dataset = torch.tensor(resnet_dataset, dtype=torch.float32)

                    video_finished = False
                    firstIteration = True
                    pos = 0
                except: 
                    continue
            else:
                try:
                    l_sample = l_dataset[pos:pos+batchsize]
                    if l_sample.shape[0] < batchsize:
                        raise StopIteration
                    hist_sample = hist_dataset[pos:pos+batchsize]
                    resnet_sample = resnet_dataset[pos:pos+batchsize]

                    l_sample = l_sample.unsqueeze(dim=0)
                    hist_sample = hist_sample.unsqueeze(dim=0)
                    resnet_sample = resnet_sample.unsqueeze(dim=0)
                    resnet_sample = resnet_sample.permute(0,1,4,2,3)
                    firstIteration_tensor = torch.zeros(1,dtype=torch.bool)


                    if firstIteration:
                        firstIteration_tensor = torch.ones(1, dtype=torch.bool)
                        firstIteration = False
                    processed.put([l_sample, firstIteration_tensor, resnet_sample, hist_sample])
                except Exception as e:
                    logger.info("Video finished in: " + str(id) + " Exception: " + str(e))
                    video_finished = True

                # get next sample from video
        logger.info("Finished. No more videos to process")

def collate(to_process, processed, video_queue, batchsize):
    logger = torch.multiprocessing.log_to_stderr()


    while not video_queue.empty():
        
        # try to collect from to_process
        samples = []
        try:
            for i in range(batchsize):
                samples.append(to_process[i].get(60))
        except Exception as e:
            return
        items = [x[0] for x in samples]
        first_it = [x[1] for x in samples]
        resnet_in = [x[2] for x in samples]
        histograms = [x[3] for x in samples]
        batch = torch.cat(items, dim=0).squeeze()
        #if len(batch.shape) == 4:
        #    batch = batch.unsqueeze(dim=1)
        resnet_in = torch.cat(resnet_in, dim=0).squeeze()
        if len(resnet_in.shape) == 4:
            resnet_in = resnet_in.unsqueeze(dim=0)
        resnet_in =  resnet_in.reshape((resnet_in.shape[0] * resnet_in.shape[1], resnet_in.shape[2], resnet_in.shape[3], resnet_in.shape[4]))
        first_it = torch.cat(first_it, dim=0).squeeze()
        histograms = torch.cat(histograms, dim=0).squeeze()
        batch.pin_memory()
        first_it.pin_memory()
        resnet_in.pin_memory()
        histograms.pin_memory()

        processed.put([batch, first_it, resnet_in, histograms])
    logger.log("VideoQueue for collate is empty")
            



class MultiProcDataset(torch.utils.data.IterableDataset):
    # videos is a multiprocessing-queue
    def __init__(self, videos, resolution=(1920,1080), batchsize=8, sequence_length=16):
        self.videos = videos
        self.manager = torch.multiprocessing.Manager()
        self.batchsize=batchsize
        self.pool = torch.multiprocessing.Pool(self.batchsize)
        self.prefetch_queue = self.manager.Queue(maxsize=4)
        logger = torch.multiprocessing.log_to_stderr()
        logger.setLevel(logging.INFO)
        logger.warning("Test")

        
        self.resolution_x, self.resolution_y = resolution
        
        self.processes = []
        self.processed = [self.manager.Queue(maxsize=4) for x in range(self.batchsize)]
        self.sequence_length = sequence_length
        self.process_handles=[]
        
    def __next__(self):
        while not self.videoQueue.empty():
            try:
                batch = self.prefetch_queue.get(10)
                return batch
            except:
                self.pool.close()
                self.pool.join()
                raise StopIteration
        raise StopIteration
    
    def __iter__(self):
        random.shuffle(self.videos)
        self.videoQueue = self.manager.Queue()
        for video in self.videos:
            self.videoQueue.put(video)
        for i in range(self.batchsize):
            processed_queue = self.processed[i]
            p = torch.multiprocessing.Process(target=do_job,args= 
                                    (i,
                                    self.videoQueue,
                                    processed_queue, 
                                    self.sequence_length))
            p.start()
            self.process_handles.append(p)

        prefetch_proc = torch.multiprocessing.Process(target=collate,args= 
                        (self.processed,
                        self.prefetch_queue,
                        self.videoQueue,
                        self.batchsize))

        prefetch_proc.start()
        self.process_handles.append(prefetch_proc)
           
        return self
  

