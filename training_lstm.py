import torch
import os
import random
from model.model_regression_lstm import ColorCNN_REGRESSION_LSTM
from model.model_classification_lstm import ColorCNN_CLASS_LSTM
from model.model_classification import ColorCNN_CLASS
from model.model_regression import ColorCNN_REGRESSION
from model.model_classification_lstm_stateful import ColorCNN_CLASS_LSTM_STATEFUL
from model.model_regression_lstm_stateful import ColorCNN_REGRESSION_LSTM_STATEFUL
import glob, os
from torch.utils.tensorboard import SummaryWriter
import cv2
from dataset.imagenet_dataset import ImageDataset
import string
from torchvision import transforms
import torchvision
import numpy as np
import torch.distributed as dist
from dataset.video_dataset import VideoDataset
from tqdm import tqdm
from skvideo import io
import skvideo
from dataset.FastDataLoader import FastDataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
from conversion_utils import *
import logging

def train(model, loader, criterion, optimizer, writer, stepsTilNow=0, rank=0, stop_after=-1, stateful=False):
    model.train()
    step = stepsTilNow
    sub_step = 0
    if stop_after == -1:
        stop_after = 99999999999
    for batch in tqdm(loader):
        frames = batch[0].cuda(non_blocking=True)
        first_frame = batch[1]
        l_in = frames[:,0].unsqueeze(dim=1)
        ab_target = frames[:,1:3]

        if stateful:
            if first_frame:
                model.reset_hidden_states()
        model_out = model(l_in)
        
        loss = criterion(model_out, ab_target)

        if rank == 0:
            writer.add_scalar('LOSS', loss, step)
    
        optimizer.zero_grad()
        loss.backward() 
        
        optimizer.step()
        stepsTilNow += 1
        step += 1
        sub_step +=1
        if sub_step > stop_after:
            break
    if stateful:
        model.reset_hidden_states()
    return step


def train_classification(model, loader, criterion, weights_a, weights_b, optimizer, writer, stepsTilNow=0, rank=0, stop_after=-1, stateful=False):
    model.train()
    step = stepsTilNow
    substep = 0
    if stop_after == -1:
        stop_after = 999999999999
    for batch in tqdm(loader):
        frames = batch[0].cuda(non_blocking=True)
        a_indices = batch[1].cuda(non_blocking=True).squeeze()
        b_indices = batch[2].cuda(non_blocking=True).squeeze()
        l_in = frames[:,0].unsqueeze(dim=1)
        first_frame = batch[3]

        if stateful == True:
            if first_frame:
                model.reset_hidden_states()
        a_out, b_out = model(l_in)
        a_out = a_out.reshape(a_out.shape[0], a_out.shape[1], a_out.shape[2]* a_out.shape[3])
        b_out = b_out.reshape(b_out.shape[0], b_out.shape[1], b_out.shape[2]* b_out.shape[3])
        loss_a = criterion(a_out, a_indices, weights_a)
        loss_b = criterion(b_out, b_indices, weights_b)
        loss = (loss_a + loss_b) / 2.0

        if rank == 0:
            writer.add_scalar('LOSS', loss.item(), step)
        optimizer.zero_grad()

        loss.backward() 
        optimizer.step()
        stepsTilNow += 1
        step += 1
        substep += 1
        if substep > stop_after:
            break
    if stateful == True:
        model.reset_hidden_states()
    return step

def custom_loss(out, target, weights):
    target = target.permute(0,2,1)
    argmax_z = torch.argmax(target, dim=1)
    weights = weights[argmax_z].unsqueeze(dim=1)
    weighted_target = target * weights
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(-weighted_target * logsoftmax(out), dim=1))


def evaluate_classification(model, epoch, test_loader, writer):
    with torch.no_grad():
        model.eval()
        images = []
        for batch in test_loader:
            frames = batch[0].cuda(non_blocking=True)
            l_in = frames[:,0].unsqueeze(dim=1)

            a_out, b_out = model(l_in)
            a_out = a_out.permute(0,2,3,1)
            b_out = b_out.permute(0,2,3,1)

            a = distribution_to_color(a_out)
            b = distribution_to_color(b_out)
            a = a.unsqueeze(dim=1)
            b = b.unsqueeze(dim=1)
            combined = torch.cat((l_in, a, b), dim=1)
            combined = combined.cpu()
            combined = combined.permute(0,2,3,1)
            combined = combined.numpy()
            rgb = np.zeros(combined.shape, dtype=np.uint8)
            for i in range(combined.shape[0]):
                frame = combined[i]
                frame *= 255
                frame = np.clip(frame, 1, 254)
                frame = frame.astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
                rgb[i] = 255 - frame
            rgb = torch.tensor(rgb)
            rgb = rgb.permute(0,3,1,2)
            writer.add_images("Images", rgb, global_step=epoch)
            break


def evaluate(model, epoch,test_loader, writer):
    with torch.no_grad():
        model.eval()
        images = []
        for batch in test_loader:
            frames = batch[0].cuda(non_blocking=True)
            l_in = frames[:,0].unsqueeze(dim=1)
            model_out = model(l_in)
            model_out *= 127
            l_in *= 100

            combined = torch.cat((l_in, model_out), dim=1)
            combined = combined.cpu()
            combined = combined.permute(0,2,3,1)
            combined = combined.numpy()
            rgb = np.zeros(combined.shape, dtype=np.uint8)
            for i in range(combined.shape[0]):
                frame = combined[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
                frame *= 255
                frame = frame.astype(np.uint8)
                rgb[i] = 255 -frame
            rgb = torch.tensor(rgb)
            rgb = rgb.permute(0,3,1,2)
            writer.add_images("Images", rgb, global_step=epoch)
            break
    
def loadVideos(path):
    # load all .mp4-files from path into list
    train_list = path + "/train_filenames.txt"
    test_list  =path + "/test_filenames.txt"

    f = open(train_list, "r")
    training_names = f.readlines()
    training_names = [x.strip() for x in training_names]
    #training_names = [os.path.basename(item) for item in training_names]
    f.close()

    f = open(test_list, "r")
    test_names = f.readlines()
    test_names = [x.strip() for x in test_names]
    f.close()
    return training_names, test_names



def video_evaluation_regression(model, epoch, save_path, video_to_load, use_lstm=True, stateful=False, imagenet=False):
    filename = video_to_load
    data = skvideo.io.ffprobe(filename)['video']
    rate = data['@r_frame_rate']
    stepsize = 32
    if use_lstm:
        if stateful:
            foutname = save_path + str(epoch) + "_lstm_stateful_regression_eval.mp4"
        else:
            foutname = save_path + "" +str(epoch) + "_lstm_regression_eval.mp4"
        print("Saved result-evaluation-video as: " + foutname)
    else:
        if imagenet:
            foutname = save_path + "" +str(epoch) + "_simplecnn_imagenet_regression_eval.mp4"
        else:
            foutname = save_path + "" +str(epoch) + "_simplecnn_regression_eval.mp4"
        print("Saved result-evaluation-video as: " + foutname)
    vid_out = skvideo.io.FFmpegWriter(foutname, inputdict={
        '-r': rate,
        }, outputdict={'-b':'5M'})
    videoreader = iter(skvideo.io.vreader(filename))
    should_read = True
    finished = 0
    while should_read:
        frames = []
        try:
            for i in range(stepsize):
                frames.append(next(videoreader))
        except:
            should_read = False
        with torch.no_grad():
            frames_np = np.stack(frames)
            model.eval()
            frames = frames_to_lab(frames_np)
            l_in = frames[:,0].unsqueeze(dim=1).cuda()
            model_out = model(l_in)
            combined = torch.cat((l_in, model_out), dim=1)
            combined = combined.cpu()
            combined = combined.permute(0,2,3,1)
            combined = combined.numpy()
            rgb = frames_to_rgb(combined)
            for frame in rgb:
                vid_out.writeFrame(frame)
            finished += stepsize
            print("Finished: ", finished, " Frames")
    vid_out.close()

def video_evaluation_classification(model, epoch, path, video_to_load, use_lstm=True, stateful=False, imagenet=False):
    filename = video_to_load
    data = skvideo.io.ffprobe(filename)['video']
    rate = data['@r_frame_rate']
    stepsize = 24
    if use_lstm:
        if stateful:
            foutname = path + str(epoch) + "_lstm_stateful_classification_eval.mp4"
        else:
            foutname = path + "" +str(epoch) + "_lstm_classification_eval.mp4"
    else:
        if imagenet:
            foutname = path + "" +str(epoch) + "_simplecnn_classification_imagenet_eval.mp4"
        else:
            foutname = path + "" +str(epoch) + "_simplecnn_classification_eval.mp4"
    vid_out = skvideo.io.FFmpegWriter(foutname, inputdict={
        '-r': rate,
        }, outputdict={'-b':'5M'})
    videoreader = iter(skvideo.io.vreader(filename))
    should_read = True
    finished = 0
    while should_read:
        frames = []
        try:
            for i in range(stepsize):
                frames.append(next(videoreader))
        except:
            should_read = False
        with torch.no_grad():
            model.eval()
            frames_np = np.stack(frames)
            frames = frames_to_lab(frames_np)
            l_in = frames[:,0].unsqueeze(dim=1).cuda()
            a_out, b_out = model(l_in)
            a_out = a_out.permute(0,2,3,1)
            b_out = b_out.permute(0,2,3,1)
            a = distribution_to_color(a_out)
            b = distribution_to_color(b_out)
            a *= 2
            a -= 1
            b *= 2
            b -= 1
            a = a.unsqueeze(dim=1)
            b = b.unsqueeze(dim=1)
            combined = torch.cat((l_in, a, b), dim=1)
            combined = combined.cpu()
            combined = combined.permute(0,2,3,1)
            combined = combined.numpy()

            rgb = frames_to_rgb(combined)
            for frame in rgb:
                vid_out.writeFrame(frame)
            finished += stepsize
            print("Finished: ", finished, " Frames")
    vid_out.close()

def main():
    parser = argparse.ArgumentParser()
    log = logging.getLogger("my-logger")
    log.info("Hello, world")
    log.info("Started Script")

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--trainingtype", default="regression")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--eval_savepath", default="/network-ceph/pgrundmann/video_evaluations/")
    parser.add_argument("--experiment_name", default="lstm")
    parser.add_argument("--testvideo_filename", default="/network-ceph/pgrundmann/sony.mp4")
    parser.add_argument("--steps_per_epoch", default=-1, type=int)    
    parser.add_argument("--no_lstm", default=False, action='store_true')
    parser.add_argument("--stateful", default=False, action='store_true')    
    parser.add_argument("--imagenet", default=False, action='store_true')
    args = parser.parse_args()
    log.info(str(args))

    if bool(args.no_lstm):
        print("No LSTM")
    else:
        print("Use LSTM")
    args.use_lstm = not args.no_lstm

    torch.multiprocessing.set_start_method('spawn')
    config_name = args.experiment_name
    writer = SummaryWriter(log_dir="/network-ceph/pgrundmann/video_runs/" +config_name+"/")
    
    BATCH_SIZE=args.batchsize
    EPOCHS=args.epochs
    LR = 0.0001
    TRAIN_MODE = args.trainingtype

    if args.imagenet:
        composed = transforms.Compose([transforms.Resize(128),
                               transforms.RandomCrop(112)])
        imagenet_train = torchvision.datasets.ImageFolder("/network-ceph/pgrundmann/maschinelles_sehen/ImageNet-Datasets-Downloader/imagenet/imagenet_images", transform=composed)
        imnet_ds = ImageDataset(imagenet_train, type='training', training=TRAIN_MODE)
        if TRAIN_MODE == 'classification':
            weights_a = torch.tensor(imnet_ds.weights_a, dtype=torch.float).cuda()
            weights_b = torch.tensor(imnet_ds.weights_b, dtype=torch.float).cuda()
        loader = FastDataLoader(imnet_ds, shuffle=True, batch_size=args.batchsize, pin_memory=True, num_workers=8)
    else:
        train_video_filenames, _ = loadVideos("/network-ceph/pgrundmann/youtube_processed")
        print("Loaded Video-filenames")
        train_loaders = []
        train_datasets = []
        random.shuffle(train_video_filenames)
        try:
            train_datasets = torch.load("/network-ceph/pgrundmann/maschinelles_sehen/train_datasets.pt")
        except:
            for fname in tqdm(train_video_filenames):
                train_datasets.append(VideoDataset(fname, BATCH_SIZE))
            torch.save(train_datasets, "train_datasets.pt")   

        for ds in train_datasets:
            ds.training_type = TRAIN_MODE
        if TRAIN_MODE == 'classification':
            # calculate the weights for a and b based on imagenet (is faster than to do it on the video-dataset)
            composed = transforms.Compose([transforms.Resize(128),
                                transforms.RandomCrop(112)])
            imagenet_train = torchvision.datasets.ImageFolder("/network-ceph/pgrundmann/maschinelles_sehen/ImageNet-Datasets-Downloader/imagenet/imagenet_images", transform=composed)
            imnet_ds = ImageDataset(imagenet_train, type='training', training=TRAIN_MODE)
            weights_a = torch.tensor(imnet_ds.weights_a, dtype=torch.float).cuda()
            weights_b = torch.tensor(imnet_ds.weights_b, dtype=torch.float).cuda()
        
        ds = torch.utils.data.ConcatDataset(train_datasets)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True)
    
    print("Loaded dataloaders and datasets")

    if TRAIN_MODE == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        criterion = custom_loss
    
    if TRAIN_MODE == 'classification':
        if args.use_lstm:
            if args.stateful:
                model = ColorCNN_CLASS_LSTM_STATEFUL()
            else:
                model = ColorCNN_CLASS_LSTM()
        else:
            model = ColorCNN_CLASS()
    else:
        if args.use_lstm:
            if args.stateful:
                model = ColorCNN_REGRESSION_LSTM_STATEFUL()
            else:
                model = ColorCNN_REGRESSION_LSTM()
        else:
            model = ColorCNN_REGRESSION()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-8)
    
    step = 0
    for i in range(EPOCHS):
        if TRAIN_MODE == 'classification':
            step = train_classification(model, loader, criterion, weights_a, weights_b, optimizer, writer,stepsTilNow=step,stop_after=args.steps_per_epoch, stateful=args.stateful)
            video_evaluation_classification(model,i,args.eval_savepath, args.testvideo_filename, use_lstm=args.use_lstm,imagenet=args.imagenet)
            if not args.imagenet:
                random.shuffle(train_datasets)
                ds = torch.utils.data.ConcatDataset(train_datasets)
                loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True)
            if args.use_lstm:
                torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "/network-ceph/pgrundmann/video_models/lstm_classification_" + str(i) + ".bin")
            else:
                torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "/network-ceph/pgrundmann/video_models/simple_cnn_classification_" + str(i) + ".bin")
        else:
            step = train(model, loader, criterion, optimizer, writer, stepsTilNow=step, stop_after=args.steps_per_epoch, stateful=args.stateful)
            video_evaluation_regression(model,i,args.eval_savepath, args.testvideo_filename, use_lstm=args.use_lstm, imagenet=args.imagenet)
            if not args.imagenet:
                random.shuffle(train_datasets)
                ds = torch.utils.data.ConcatDataset(train_datasets)
                loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, pin_memory=True)
            if args.use_lstm:
                torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "/network-ceph/pgrundmann/video_models/lstm_regression_" + str(i) + ".bin")
            else:
                torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict()}, "/network-ceph/pgrundmann/video_models/simple_cnn_regression_" + str(i) + ".bin")
        
        
if __name__ == "__main__":

    main()




