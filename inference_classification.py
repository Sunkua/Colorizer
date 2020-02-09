import torch
import os
import random
from model.model_interpol import ColorCNN
import glob, os
import argparse
from torch.utils.tensorboard import SummaryWriter
import skvideo
from dataset.multiproc_dataset import MultiProcDataset
import string
import numpy as np
import cv2
from tqdm import tqdm
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from apex import amp
from apex.parallel import DistributedDataParallel
from torchvision import transforms
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

def main():
   
    SAVE_AFTER = 512
    SKIP = 1024
    torch.multiprocessing.set_start_method('spawn')

    _, testvideos = loadVideos("/network-ceph/pgrundmann/youtube_processed")

    filename = testvideos[0] #'/network-ceph/pgrundmann/youtube_eval/opalaxy_proc.mp4' # testvideos[0]


    vid_in = skvideo.io.FFmpegReader(filename)
    data = skvideo.io.ffprobe(filename)['video']
    rate = data['@r_frame_rate']
    T = np.int(data['@nb_frames'])

    videoreader = iter(skvideo.io.vreader(filename))
   
    model = ColorCNN()
    model = model.cuda()

    checkpoint = torch.load('/network-ceph/pgrundmann/video_model_gru_steps_33000.bin')
    model.load_state_dict(checkpoint)

    model = model.cuda()
    model.h_n = None
    model.c_n = None

    vid_out = skvideo.io.FFmpegWriter("testvideo_color.mp4", inputdict={
        '-r': rate,
        }, outputdict={'-b':'5M'})
    '''
    cnt = 0
    for frame in videoreader:
        if cnt < 255:
            cnt += 1
            continue
        frame = frame.astype(np.float32) / 255.0
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        frame_newrgb = cv2.cvtColor(frame_lab, cv2.COLOR_Lab2RGB) * 255
        frame_newrgb = frame_newrgb.astype(np.uint8)
        vid_out.writeFrame(frame_newrgb)
        cnt += 1
        if cnt > 512:
            break
    return

    '''

    should_read = True
    stepsize = 16
    first_it = torch.ones(1, dtype=torch.bool) 
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
        l = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.float32)
        resnet_input = np.zeros(frames.shape, dtype=np.float32)
        for i in range(frames.shape[0]):
            frame = frames[i] 
            #frame = frame.astype(np.float32)
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
            frame_lab = frame_lab.astype(np.float32)
            frame_lab[:,:,0] /= 255
            frame_lab[:,:,0] -= 0.5
            resnet_in = np.expand_dims(frame_lab[:,:,0],axis=2)
            resnet_in = np.repeat(resnet_in,3, axis=2)
            resnet_input[i] = resnet_in
            l[i] = frame_lab[:,:,0]

        
        l = torch.tensor(l, dtype=torch.float32)
        to_process = l.unsqueeze(dim=0)
        if len(to_process.shape) < 4:
            to_process = to_process.unsqueeze(dim=0)
        to_process = to_process.unsqueeze(dim=4)
        resnet_in = torch.tensor(resnet_input, dtype=torch.float32)
        resnet_in = resnet_in.permute((0,3,1,2))
        
        # predict
        model.eval()
        with torch.no_grad():
            model_in = to_process.cuda(non_blocking=True)
            resnet_in = resnet_in.cuda(non_blocking=True)
            for i in range(resnet_in.shape[0]):
                resnet_in[i] = transform(resnet_in[i])

            out = model(model_in, first_it, resnet_in)
            out = out.cpu().squeeze()
            first_it[0] = 0
            if len(out.shape) < 4:
                out = out.unsqueeze(dim=0)
            out = out.permute(0,2,3,1)
            to_process = to_process.squeeze()
            to_process = to_process.unsqueeze(dim=3)
            
            res = torch.cat((to_process, out), dim=3)
            res = res.numpy()

            for i in range(res.shape[0]):
                frame_ = res[i]
                frame_[:,:,0] += 0.5
                frame_ *= 255
                #frame_[:,:,0] *= 255
                #frame_[:,:,1:3] = np.interp(frame[:,:,1:3], (0, 1),(0,255))
                in_frame = frame_.astype(np.uint8)
                frame_rgb = cv2.cvtColor(in_frame, cv2.COLOR_Lab2RGB)
                vid_out.writeFrame(frame_rgb)

        frame_cnt += stepsize
        print(str(frame_cnt) + " of " + str(T) + " Frames processed!")
        if frame_cnt > SAVE_AFTER:
            break
        
    
    vid_out.close()

if __name__ == "__main__":
    main()





