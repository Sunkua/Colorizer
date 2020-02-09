import torch
import numpy as np
import torchvision
import cv2
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import randrange

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, type='training', training='regression'):
        #self.dataset = dataset
        self.dataset = dataset
        self.length = len(dataset)
        '''
        if training == 'regression':
            self.featurecache = {}
            model = models.resnext50_32x4d(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1]) 
            self.model = model
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.cuda()
        '''
        self.training=training
        stop = 1 + ((1.0 / 15) / 2)
        self.bins = np.arange(start=0,stop=1,step=(1.0/15))
        self.bins += ((1 / 15.0) / 2.0)
        self.type = type
        if self.type == 'training' and self.training == 'classification':
            self.calculate_weights(dataset)
        
        #self.bins = self.bins.reshape(-1,1)
        #self.nbrs = NearestNeighbors(n_neighbors=5).fit(self.bins)

    def calculate_weights(self, dataset):
        weights_a = None
        weights_b = None
        cnt = 0
        for i in tqdm(range(len(dataset))):
            idx = randrange(len(dataset))
            image = dataset[idx]
            image = np.array(image[0])
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            frame = frame.astype(np.float32)
            frame /= 255.0
            a = frame[:,:,1]
            b = frame[:,:,2]
            a_hist = np.histogram(a, bins=15, range=(0,1), density=True)[0]
            b_hist = np.histogram(b, bins=15, range=(0,1), density=True)[0]
            if weights_a is None:
                weights_a = a_hist
                weights_b = b_hist
            else:
                weights_a += a_hist
                weights_b += b_hist
                weights_a /= 2.0
                weights_b /= 2.0
            cnt += 1
            if cnt > 5000:
                break

        weights_a[weights_a==0.0] = 1.0
        weights_b[weights_b==0.0] = 1.0
        weights_a = weights_a / np.sum(weights_a)
        weights_b = weights_b / np.sum(weights_b)
        smoothed_a = gaussian_filter(weights_a, sigma=5)
        smoothed_b = gaussian_filter(weights_b, sigma=5)
        smoothed_a = weights_a / np.sum(weights_a)
        smoothed_b = weights_b / np.sum(weights_b)
        #smoothed_mixed_a = (0.5 * smoothed_a) + (0.5 / (weights_a.shape[0]))
        #smoothed_mixed_b = (0.5 * smoothed_b) + (0.5 / (weights_b.shape[0]))
        smoothed_a = 1 - smoothed_a
        smoothed_b = 1 - smoothed_b
        smoothed_a = smoothed_a / np.sum(smoothed_a)
        smoothed_b = smoothed_b / np.sum(smoothed_b)
        self.weights_a = smoothed_a
        self.weights_b = smoothed_b


    def get_classification_data(self,idx):
        if self.dataset is None:
            if self.type == 'training':
                    composed = transforms.Compose([transforms.Resize(128),
                               transforms.RandomCrop(112)])
                    self.dataset = torchvision.datasets.ImageFolder("/network-ceph/pgrundmann/maschinelles_sehen/ImageNet-Datasets-Downloader/imagenet/imagenet_images", transform=composed)
            else:
                    composed = transforms.Compose([transforms.Resize(128),
                               transforms.RandomCrop(112)])
                    self.dataset = torchvision.datasets.ImageFolder("/network-ceph/pgrundmann/maschinelles_sehen/ImageNet-Datasets-Downloader/imagenet/imagenet_images", transform=composed)
        image = self.dataset[idx][0]
        image = np.array(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        frame = frame.astype(np.float32)
        frame /= 255.0
        a = frame[:,:,1]
        b = frame[:,:,2]
        a = a.reshape(a.shape[0] * a.shape[1],1)
        b = b.reshape(b.shape[0] * b.shape[1],1)

        indices_a = np.searchsorted(self.bins, a)
        indices_b = np.searchsorted(self.bins, b)
        
        indices_a = torch.tensor(indices_a, dtype=torch.long)
        indices_b = torch.tensor(indices_b, dtype=torch.long)

        target_a = torch.nn.functional.one_hot(indices_a, 15)
        target_b = torch.nn.functional.one_hot(indices_b, 15)
        target_a = torch.tensor(target_a, dtype=torch.float)
        target_b = torch.tensor(target_b, dtype=torch.float)

        frame = torch.tensor(frame, dtype=torch.float32)   
        frame = frame.permute(2,0,1)
        first_frame = torch.ones(1, dtype=torch.bool)
        return [frame, target_a, target_b, first_frame]
        
    def get_regression_data(self, idx):
        image = self.dataset[idx][0]
 
        image = np.array(image)
        image = image.astype(np.float32)
        image /= 255.0
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        frame[:,:,0] /= 100
        frame[:,:,1:3] /= 127.0 
       
        frame = torch.tensor(frame, dtype=torch.float32)   
        frame = frame.permute(2,0,1)
        first_frame = torch.ones(1, dtype=torch.bool)

        '''
        if idx in self.featurecache:
            resnext_feature = self.featurecache[idx]
        else:
            resnet_in = frame[0].unsqueeze(dim=0).cuda()
            resnet_in = resnet_in.unsqueeze(dim=0)
            scale_factor = 224 / resnet_in.shape[2] 
            resnet_in = torch.nn.functional.interpolate(resnet_in, scale_factor=scale_factor)
            resnet_in = resnet_in.repeat((1,3,1,1))
            deeplab_out = self.model(resnet_in)
            deeplab_out = deeplab_out.cpu().squeeze()
            deeplab_out = deeplab_out.unsqueeze(dim=1)
            deeplab_out =  deeplab_out.unsqueeze(dim=2)
            self.featurecache[idx] = deeplab_out
            resnext_feature = deeplab_out
        '''
        return [frame, first_frame]


    def save_features(self):
        torch.save(self.featurecache, "resnext-features.pt")
    
    def load_features(self):
        self.featurecache = torch.load("resnext-features.pt")
        

    def __getitem__(self, idx):
        if self.training == 'regression':
            return self.get_regression_data(idx)
        else:
            return self.get_classification_data(idx)
        

    def __len__(self):
        return self.length

