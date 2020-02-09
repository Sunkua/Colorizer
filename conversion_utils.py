import cv2
import torch
import numpy as np

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

def distribution_to_color(distribution, T = 0.38):
    start = 0
    end = 1
    step = 1.0 / 15.0
    quantized = torch.arange(start=start,end=end,step=step, dtype=torch.float, device='cuda')

    z = torch.exp(torch.log(distribution + 1e-8) / T)
    z = z / torch.sum(z, dim=3, keepdim=True)
    z = torch.sum(z * quantized, dim=3)
    return z