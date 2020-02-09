import torch
import numpy as np
import torchvision
import cv2
import torchvision.models as models
import skvideo
from skvideo import io


def frames_to_lab(frames):
    frames_lab = np.zeros(frames.shape, dtype=np.float32)
    for i, frame in enumerate(frames):
        frame = frame.astype(np.float32)
        frame /= 255.0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        frame[:,:,0] /= 100.0
        frame[:,:,1:3] /= 127.0
        frames_lab[i] = frame
    lab_tensor = torch.tensor(frames_lab, dtype=torch.float)
    lab_tensor = lab_tensor.permute(0,3,1,2)
    return lab_tensor


# takes x frames of lab images in fp32 format normalized between 0 and 1
# Format num_frames, height, width, color_channel 
# return rgb uint8 frames 

def frames_to_rgb(frames):
    frames_rgb = np.zeros(frames.shape, dtype=np.uint8)
    for i, frame in enumerate(frames):
        frame[:,:,0] *= 100
        frame[:,:,1:3] *= 127
        frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
        frame *= 255.0
        frame = frame.astype(np.uint8)
        frames_rgb[i] = frame
    return frames_rgb


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, video, batchsize, training_type='regression'):
        data = skvideo.io.ffprobe(video)['video']
        self.length = np.int(data['@nb_frames']) // batchsize
        self.video = video
        self.videofile = None
        self.batchsize = batchsize
        self.training_type = training_type
        self.bins = np.arange(start=0,stop=1,step=(1.0/15))
        self.bins += ((1 / 15.0) / 2.0)

    def __get_classification_data__(self,idx):
        frames = self.videofile[idx*self.batchsize:(idx+1)*self.batchsize]
        frames = frames_to_lab(frames)
        frames = frames.permute(0,2,3,1)
        frames = frames.numpy()
        
        a = frames[:,:,:,1]
        b = frames[:,:,:,2]
        a += 1
        b += 1
        a /= 2.0
        b /= 2.0
        a = a.reshape(a.shape[0] * a.shape[1] * a.shape[2],1)
        b = b.reshape(b.shape[0] * b.shape[1] * b.shape[2],1)

        indices_a = np.searchsorted(self.bins, a)
        indices_b = np.searchsorted(self.bins, b)
        
        indices_a = torch.tensor(indices_a, dtype=torch.long)
        indices_b = torch.tensor(indices_b, dtype=torch.long)

        target_a = torch.nn.functional.one_hot(indices_a, 15)
        target_b = torch.nn.functional.one_hot(indices_b, 15)
        target_a = torch.tensor(target_a, dtype=torch.float)
        target_b = torch.tensor(target_b, dtype=torch.float)
        target_a = target_a.reshape(frames.shape[0], frames.shape[1] * frames.shape[2],15)
        target_b = target_b.reshape(frames.shape[0], frames.shape[1] * frames.shape[2],15)
        frames = torch.tensor(frames)
        frames = frames.permute(0,3,1,2)
        first_frame = torch.zeros(1, dtype=torch.bool)
        if idx == 0:
            first_frame = torch.ones(1, dtype=torch.bool)

        return [frames, target_a, target_b, first_frame]

    def __get_regression_data__(self,idx):
        frames = self.videofile[idx*self.batchsize:(idx+1)*self.batchsize]
        frames = frames_to_lab(frames)
        first_frame = torch.zeros(1, dtype=torch.bool)
        if idx == 0:
            first_frame = torch.ones(1, dtype=torch.bool)
        return [frames, first_frame]


    def __getitem__(self, idx):
        if self.videofile is None:
            self.videofile = skvideo.io.vread(self.video)
        if self.training_type == 'regression':
            return self.__get_regression_data__(idx)
        else:
            return self.__get_classification_data__(idx)

            

    def __len__(self):
        return self.length

