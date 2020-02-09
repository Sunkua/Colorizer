import torch
import random
from skimage import color
import numpy as np
import torchvision
import cv2
from itertools import chain, cycle

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video, resolution=(1920,1080), batchsize=16):
        self.video = video
        self.resolution_x, self.resolution_y = resolution
        self.position = 0
        self.batchsize=batchsize
        
        
    def resizeAndPad(self, img, size, padColor=0):
        h, w = img.shape[:2]
        sh, sw = size

        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

    def __next__(self):
        sample = torch.empty(0)
        for i in range(self.batchsize):
            if (self.position < self.__len__()):
                ret, frame = self.vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                
                frame = self.resizeAndPad(frame,(640,360))
                frame = cv2.normalize(frame, 0, 255, cv2.NORM_MINMAX)
                
                self.position += 1
                tensor= torch.tensor(frame, dtype=torch.float32).unsqueeze(dim=0)
                sample = torch.cat((sample, tensor), dim=0)
            else:
                self.vid.release()
                raise StopIteration
        return sample
            

    def __iter__(self):
        self.vid = cv2.VideoCapture(self.video)
        self.len = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        return self
        #return iter(self.__next__() for _ in range(self.__len__()))

    
    def __getitem__(self, idx):
        if (self.position < self.__len__()):
            ret, frame = self.vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            

            frame = self.resizeAndPad(frame,(1920,1080))
            frame = cv2.normalize(frame, 0, 255, cv2.NORM_MINMAX)
            
            self.position += 1
            tensor= torch.tensor(frame, dtype=torch.float32)
            return tensor
        else:
            self.vid.release()
            return None
    
    def __len__(self):
        return self.len


   
