import torch
import os
import random
from model.model_interpol import ColorCNN 
import glob, os
from torch.utils.tensorboard import SummaryWriter
import cv2
from dataset.multiproc_dataset import MultiProcDataset
import string
from torchvision import transforms
from apex import amp
from apex.parallel import DistributedDataParallel
import argparse
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

def train(model, train_loader, criterion, optimizer, writer, rank=1, stepsTilNow=0):
    if rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name, param, -1) 

    model.train()
    step = 0
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
    for batch in train_loader:
        first_it = batch[1]
        resnet_in = batch[2]
        batch = batch[0]
    
        resnet_in = resnet_in.cuda(non_blocking=True)
        for i in range(resnet_in.shape[0]):
            resnet_in[i] = transform(resnet_in[i])

        


        x_in = batch[:,:,:,:,0].unsqueeze(dim=4).cuda(non_blocking=True)


        # y_out_shape (batchsize, sequence_length, height, width, num_channels)

        y_out_a = batch[:,:,:,:,1].unsqueeze(dim=4).permute(0,1,4,2,3).cuda(non_blocking=True)
        y_out_b = batch[:,:,:,:,2].unsqueeze(dim=4).permute(0,1,4,2,3).cuda(non_blocking=True)

        model_out = model(x_in, first_it, resnet_in)
        # out (batchsize, sequencelength, num_channels, height, width)

        l1 = criterion(model_out[:,:,0].unsqueeze(dim=2), y_out_a)
        l2 = criterion(model_out[:,:,1].unsqueeze(dim=2), y_out_b)
        loss = (l1 + l2) / 2
        #sequenceLength = x_in.shape[1]
        #batchsize = x_in.shape[0]
        #loss /= (sequenceLength * batchsize)
        if rank == 0:
            writer.add_scalar('LOSS', loss, step)
            if step % 100 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param, step) 
        
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
        stepsTilNow += 1
        step += 1
        if rank == 0:
            if step % 100 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }

                torch.save(checkpoint, "/network-ceph/pgrundmann/video_model_gru_steps_" + str(step) + ".bin") 

def evaluate(model, epoch,test_loader,criterion, writer):
    losses = torch.empty(0)
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            x_in = batch[0].cuda()
            y_out_a = batch[1].cuda()
            y_out_b = batch[2].cuda()
            mask = batch[3].cuda()

            model_out = model(x_in)
            model_out *= mask
            l1 = criterion(model_out(model_out[:,0], y_out_a))
            l2 = criterion(model_out(model_out[:,1], y_out_b))
            loss = (l1 + l2) / 2
            losses = torch.stack((losses,loss),dim=0)

    acc = torch.mean(losses, dim=0)
    writer.add_scalar("Accuracy",acc,epoch)

def main():
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

   
    end = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    config_name = "VIDEO-MODEL-CNN-ONLY" + end
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir="/network-ceph/pgrundmann/video_runs/" +config_name+"/",max_queue=20)
    else:
        writer = None

    BATCH_SIZE=8
    SEQUENCE_LENGTH=16
    EPOCHS=1
    LR = 0.0001
    
    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    path = '/network-ceph/pgrundmann/youtube_processed'

    training_videos, test_videos = loadVideos(path)

    dataset = MultiProcDataset(training_videos, batchsize=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=None, num_workers=0, pin_memory=True)
  
    model = ColorCNN()
    model = model.cuda()
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-8)

    opt_level = 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    if args.distributed:
        model = DistributedDataParallel(model)

    
    for i in range(EPOCHS):
        train(model,train_loader, criterion, optimizer, writer, rank=args.local_rank)
        print("Epoch finished")
        torch.save(model.state_dict(), "/network-ceph/pgrundmann/video_model_" + str(i) + ".bin")
   #     evaluate(model,(i+1), test_loader,criterion,writer)



if __name__ == "__main__":
    main()




