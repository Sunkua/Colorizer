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

# to_process: global video_filename_list
# processed: frames for the current process, processed exists for every process (cnt(batchsize))
def do_job(id, to_process, processed, batchsize):
    logger = torch.multiprocessing.log_to_stderr()
    video_finished = True
    activeVideo = None
    activeVideoReader = None
    firstIteration = True
    pos = 0
    bins_x = np.arange(start=0,stop=1,step=(1.0/17))
    bins_y = np.arange(start=0,stop=1,step=(1.0/17))
    mesh = np.dstack(np.meshgrid(bins_x, bins_y)).reshape(-1, 2)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(mesh)
    while not to_process.empty():
        if video_finished:
            # try to get next one
            try:
                logger.info(str(to_process.qsize()) + " items remaining in queue")
                activeVideo = to_process.get(10)
                logger.info("Load: " + activeVideo)
                activeVideoReader = skvideo.io.vreader(activeVideo)
                video_finished = False
                firstIteration = True
                pos = 0
            except: 
                continue
        else:
            try:
                samples = []
                for i in range(batchsize):
                    sample = next(activeVideoReader)
                    sample = np.expand_dims(sample, axis=0)
                    samples.append(sample)
                samples = np.concatenate(samples,axis=0)
                processed_samples = np.zeros((samples.shape[0], 
                                              samples.shape[1], 
                                              samples.shape[2],
                                              1),
                                              dtype=np.float32)

                sample_histograms = np.zeros((samples.shape[0], 
                                              samples.shape[1] * samples.shape[2],
                                              17*17),
                                              dtype=np.float32)
                resnet_in_all = np.zeros(samples.shape, dtype=np.float32)
                if samples.shape[0] < batchsize:
                    logger.info("Not enough samples. Before StopIteration-Exception")
                    raise StopIteration
                for i in range(batchsize):
                    frame = samples[i]
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
                    frame = frame.astype(np.float32)
                    frame /= 255.0    
                    l = frame[:,:,0]                
                    '''
                    #ab = frame[:,:,1:3]
                    l = frame[:,:,0]
                    l -= 0.5
                    frame[:,:,0] = l
                    #frame[:,:,1:3] = ab
                    '''
                    ab = frame[:,:,1:3]
                    ab = ab.reshape((ab.shape[0]*ab.shape[1], 2))

                    distances, indices = nbrs.kneighbors(ab)
                    
                    # normalize distances and create reciprocal for one-hot values
                    distances = np.power(distances, -1)
                    distances_sums = np.sum(distances, axis=1)
                    distances_sums = np.expand_dims(distances_sums, axis=1)
                    distances /= distances_sums
                    distances = np.expand_dims(distances, axis=2)
                    
                    # create "onehot"-encodings for all pixels based on their distances to the knn
                    #one_hot = np.zeros((ab.shape[0], 17*17),dtype=np.float32)
                    sample_histograms[i,indices] = distances

                    l = frame[:,:,0]
                    l = np.expand_dims(l, axis=2)
                    resnet_in = np.expand_dims(frame[:,:,0],axis=2)
                    resnet_in = np.repeat(resnet_in,3, axis=2)
                    resnet_in_all[i] = resnet_in
                    processed_samples[i] = l
                    #sample_histograms[i] = one_hot


                pos += batchsize
                tensor_l = torch.tensor(processed_samples, dtype=torch.float32).unsqueeze(dim=0)
                tensor_hists = torch.tensor(sample_histograms, dtype=torch.float32).unsqueeze(dim=0)

                resnet_in_all = torch.tensor(resnet_in_all,dtype=torch.float32).unsqueeze(dim=0)
                resnet_in_all = resnet_in_all.permute(0,1,4,2,3)
                if firstIteration:
                    first_it = torch.ones(1, dtype=torch.bool)
                else:
                    first_it = torch.zeros(1, dtype=torch.bool)
                processed.put([tensor_l, first_it, resnet_in_all, tensor_hists])
                firstIteration = False
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
        batch.pin_memory()
        resnet_in = torch.cat(resnet_in, dim=0).squeeze()
        if len(resnet_in.shape) == 4:
            resnet_in = resnet_in.unsqueeze(dim=0)
        resnet_in =  resnet_in.reshape((resnet_in.shape[0] * resnet_in.shape[1], resnet_in.shape[2], resnet_in.shape[3], resnet_in.shape[4]))
        resnet_in.pin_memory()
        first_it = torch.cat(first_it, dim=0).squeeze()
        histograms = torch.cat(histograms, dim=0).squeeze()
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
        self.prefetch_queue = self.manager.Queue(maxsize=10)
        logger = torch.multiprocessing.log_to_stderr()
        logger.setLevel(logging.INFO)
        logger.warning("Test")

        
        self.resolution_x, self.resolution_y = resolution
        
        self.processes = []
        self.processed = [self.manager.Queue(maxsize=10) for x in range(self.batchsize)]
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
  

