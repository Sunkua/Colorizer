import torch
import os
import random
from model.model_imagenet import ColorCNN 
from model.model_regression import ColorCNN_REGRESSION
from model.model_classification import ColorCNN_CLASS
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
import h5py
import hdf5plugin
from dataset.FastDataLoader import FastDataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
def train(model, loader, criterion, optimizer, writer, stepsTilNow=0, rank=0):
    model.train()
    step = stepsTilNow

    for batch in tqdm(loader):
        frames = batch[0].cuda(non_blocking=True)
        #features = batch[1].cuda(non_blocking=True)
        l_in = frames[:,0].unsqueeze(dim=1)
        ab_target = frames[:,1:3]

        model_out = model(l_in)
        loss = criterion(model_out, ab_target)

        if rank == 0:
            writer.add_scalar('LOSS', loss, step)
    
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        stepsTilNow += 1
        step += 1
    return step


def distribution_to_color(distribution, T = 0.38):
    '''
    start = 0
    end = 1
    step = 1.0 / 15.0
    z = torch.arange(start=start,end=end,step=step, dtype=torch.float, device='cuda')
    z += ((1/15.0) / 2.0)
    c = distribution * z
    c = torch.sum(c, dim=3)
    return c

    z = z.reshape(1,1,1,z.shape[0])
    z = z.repeat(distribution.shape[0],distribution.shape[1], distribution.shape[2], 1)
    '''
    start = 0
    end = 1
    step = 1.0 / 15.0
    quantized = torch.arange(start=start,end=end,step=step, dtype=torch.float, device='cuda')

    z = torch.exp(torch.log(distribution + 1e-8) / T)
    z = z / torch.sum(z, dim=3, keepdim=True)
    z = torch.sum(z * quantized, dim=3)
    return z
    '''
    z_log = torch.log(distribution) / T
    z_exp = torch.exp(z_log)
    z_q = torch.exp(torch.log(distribution) / T)
    sums = torch.sum(z_q, dim=3, keepdim=True)
    z_exp /= sums
    return torch.mean(z_exp, dim=3)
    '''
def train_classification(model, loader, criterion, weights_a, weights_b, optimizer, writer, stepsTilNow=0, rank=0):
    model.train()
    step = stepsTilNow

    for batch in tqdm(loader):
        frames = batch[0].cuda(non_blocking=True)
        a_indices = batch[1].cuda(non_blocking=True).squeeze()
        b_indices = batch[2].cuda(non_blocking=True).squeeze()
        l_in = frames[:,0].unsqueeze(dim=1)

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
        
    
    return step

def custom_loss(out, target, weights):
    target = target.permute(0,2,1)
    argmax_z = torch.argmax(target, dim=1)
    weights = weights[argmax_z].unsqueeze(dim=1)
    weighted_target = target * weights
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(-weighted_target * logsoftmax(out), dim=1))

    z_ = torch.log(out)
    target = target.permute(0,2,1)
    x = target * z_
    x = torch.sum(x, dim=1)
    argmax_z = torch.argmax(target, dim=1)
    weights = weights[argmax_z]
    x = weights * x
    x = -torch.sum(x, dim=1)
    x = torch.mean(x)
    if torch.any(torch.isnan(x)):
        x = torch.zeros((1), dtype=torch.float, device='cuda')
    return x



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
            #features = batch[1].cuda(non_blocking=True)
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
    training_names = [os.path.basename(item) for item in training_names]
    f.close()

    f = open(test_list, "r")
    test_names = f.readlines()
    test_names = [x.strip() for x in test_names]
    f.close()
    return training_names, test_names

    

def main():
    parser = argparse.ArgumentParser()
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    WORLD_SIZE = 1
    GPU_COUNT=1
    if args.distributed:
        torch.cuda.set_device(args.local_rank % GPU_COUNT)

        torch.distributed.init_process_group(backend='nccl',
                                            world_size=WORLD_SIZE,
                                            init_method='env://')


    torch.multiprocessing.set_start_method('spawn')
    end = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    config_name = "IMAGE-MODEL-CNN" + end
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir="/network-ceph/pgrundmann/video_runs/" +config_name+"/")
    else:
        writer = None

    BATCH_SIZE=256
    EPOCHS=200
    LR = 0.0002
    TRAIN_MODE = 'regression'

    

    
    train_video_filenames, test_video_filenames = loadVideos("/network-ceph/pgrundmann/youtube_processed_small")
    
    
    train_loaders = []
    train_datasets = []
    training_names = []
    with h5py.File('/network-ceph/pgrundmann/youtube_precalculated/final_dataset.hdf5', 'r') as f:
        for name in f:
            training_names.append(name)

    to_process = len(training_names) // WORLD_SIZE
    training_names = training_names[args.local_rank*to_process:(args.local_rank+1)*to_process]

    random.shuffle(training_names)
    
    for fname in tqdm(training_names):
        train_datasets.append(VideoDataset(fname, 256))
    ds = torch.utils.data.ConcatDataset(train_datasets)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=1, pin_memory=True)
    
    print("Loaded dataloaders and datasets")

    split = ''
    '''
    #stl10_dataset_train = torchvision.datasets.STL10("/network-ceph/pgrundmann/stl_10",split='train+unlabeled',download=True)
    #stl10_dataset_test = torchvision.datasets.STL10("/network-ceph/pgrundmann/stl_10",split='test',download=True)
    composed = transforms.Compose([transforms.Resize(128),
                               transforms.RandomCrop(112)])
    imagenet_train = torchvision.datasets.ImageFolder("/network-ceph/pgrundmann/maschinelles_sehen/ImageNet-Datasets-Downloader/imagenet/imagenet_images", transform=composed)
    train_length = round(len(imagenet_train) * 0.99)
    test_length = len(imagenet_train) - train_length
    train_set, val_set = torch.utils.data.random_split(imagenet_train, [train_length, test_length])
    dataset = ImageDataset(train_set,type='training', training=TRAIN_MODE)
    train_loader = FastDataLoader(dataset,batch_size=128, num_workers=12, pin_memory=True, shuffle=True)
    print("Loaded Train-Set")
    
    test_dataset = ImageDataset(val_set,type='test', training=TRAIN_MODE)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=32, num_workers=0, pin_memory=True, shuffle=False)
    print("Loaded Validation-Set")



    if TRAIN_MODE == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        weights_a = torch.tensor(dataset.weights_a, dtype=torch.float).cuda()
        weights_b = torch.tensor(dataset.weights_b, dtype=torch.float).cuda()
        criterion_a = torch.nn.CrossEntropyLoss(weight=weights_a)
        criterion_b = torch.nn.CrossEntropyLoss(weight=weights_b)
        criterion = custom_loss
    
    if TRAIN_MODE == 'classification':
        model = ColorCNN_CLASS()
    else:
        model = ColorCNN_REGRESSION()
    model = model.cuda()
    model = torch.nn.DataParallel(model)



    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-8)
    step = 0
    for i in range(EPOCHS):
        #evaluate(model, i, test_loader, writer)
        if TRAIN_MODE == 'classification':
            evaluate_classification(model, i, test_loader, writer)
            step = train_classification(model, train_loader, criterion, weights_a, weights_b, optimizer, writer,stepsTilNow=step)
        else:
            evaluate(model, i, test_loader, writer)
            step = train(model, train_loader, criterion, optimizer, writer, stepsTilNow=step)

        print("Epoch finished")
        if args.local_rank == 0:
            torch.save(model.state_dict(), "/network-ceph/pgrundmann/image_model_mixed_weights_cnn_" + str(i) + ".bin")
        
        '''
        random.shuffle(training_names)
    
        for fname in tqdm(training_names):
            train_datasets.append(VideoDataset(fname, 64))
        ds = torch.utils.data.ConcatDataset(train_datasets)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=1, pin_memory=True)
        '''
if __name__ == "__main__":

    main()




