import torch
import os
import random
from model.model_regression import ColorCNN_REGRESSION
import glob, os
import argparse
from torch.utils.tensorboard import SummaryWriter
import skvideo
import string
import numpy as np
import cv2
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from skvideo import io
def loadVideos(path):
    # load all .mp4-files from path into list
    train_list = path + "/train_filenames.txt"
    test_list  =path + "/test_filenames.txt"

    f = open(train_list, "r")
    training_names = f.readlines()
    training_names = [x.strip() for x in training_names]
    f.close()

    f = open(test_list, "r")
    test_names = f.readlines()
    test_names = [x.strip() for x in test_names]
    f.close()
    return training_names, test_names

_, testvideos = loadVideos("/network-ceph/pgrundmann/youtube_processed")
filename = '/network-ceph/pgrundmann/youtube_eval/opalaxy_proc.mp4' # testvideos[0]testvideos[0]
vid_in = skvideo.io.FFmpegReader(filename)
data = skvideo.io.ffprobe(filename)['video']
rate = data['@r_frame_rate']
T = np.int(data['@nb_frames'])

videoreader = iter(skvideo.io.vreader(filename))


model = ColorCNN_REGRESSION().cuda()
checkpoint = torch.load('/network-ceph/pgrundmann/image_model_mixed_weights_cnn_9.bin', map_location='cuda:{}'.format(0))

model_state_dict = {k[7:]: v for k, v in checkpoint.items() if 'generator' not in k}
model.load_state_dict(model_state_dict)

vid_out = skvideo.io.FFmpegWriter("testvideo_color.mp4", inputdict={
        '-r': rate,
        }, outputdict={'-b':'5M'})


SKIP = 0
SAVE_AFTER = 1024
should_read = True
stepsize = 24
frame_cnt = 0
skipped = False
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])])
while should_read:
    frames = []
    if not skipped:
        for i in range(SKIP):
            _ = next(videoreader)
        skipped = True
    try:
        for i in range(stepsize):
            frames.append(next(videoreader))
    except:
        should_read = False

    frames = np.stack(frames, axis=0)
    
    # norm and convert to lab
    l = np.empty(frames.shape, dtype=np.float32)
    for i in range(frames.shape[0]):
        frame = frames[i] 
        frame = frame.astype(np.float32)
        frame /= 255.0
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        frame_lab[:,:,0] /= 100.0
        #frame_lab = frame_lab.astype(np.float32)
        #frame_lab /= 255
        l[i] = frame_lab

    
    l = torch.tensor(l, dtype=torch.float32)
    l = l.permute(0,3,1,2)
    # predict
    model.eval()
    with torch.no_grad():
        model_in = l[:,0].unsqueeze(dim=1).cuda(non_blocking=True)
        
        out = model(model_in)
        out *= 127
        l *= 100
        out = out.cpu().squeeze()
        out = out.permute(0,2,3,1)
        l = l.permute(0,2,3,1)
        l = l[:,:,:,0].unsqueeze(dim=3)
        res = torch.cat((l, out), dim=3)
        res = res.numpy()

        for i in range(res.shape[0]):
            frame_ = res[i]
            frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_Lab2RGB)
            frame_rgb *= 255.0
            frame_rgb = frame_rgb.astype(np.uint8)
            vid_out.writeFrame(frame_rgb)

    frame_cnt += stepsize
    print(str(frame_cnt) + " of " + str(T) + " Frames processed!")
    if frame_cnt > SAVE_AFTER:
        break
    

vid_out.close()
