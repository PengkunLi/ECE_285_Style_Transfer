#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import itertools

import numpy as np
import pandas as pd 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as td
import torchvision as tv
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as modelss

import copy

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from easydict import EasyDict
import random


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[ ]:


flickr_dir = '/datasets/ee285f-public/flickr_landscape/'
wikiar_dir = '/datasets/ee285f-public/wikiart/wikiart'


# In[ ]:


opt = EasyDict()

opt.epoch = 0
opt.n_epochs = 1
opt.batchSize = 5
opt.lr = 0.0003
opt.decay_epoch = 1
opt.size = 190
opt.input_nc = 3 
opt.output_nc = 3
opt.lambda_identity = 0.5
opt.lambda_A = 10 
opt.lambda_B = 10 #back to color is given more importance
opt.cuda = device
opt.generator_A2B = 'output0/netG_A2B.pth'
opt.generator_B2A = 'output0/netG_B2A.pth'
opt.discriminator_A = 'output0/netD_A.pth'
opt.discriminator_B = 'output0/netD_B.pth'
opt.loss_dir = 'output0/'


# In[ ]:


netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device) 
netD_A = Discriminator(opt.input_nc).to(device) 
netD_B = Discriminator(opt.output_nc).to(device) 

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


# In[ ]:


class CycleGanDatasets(td.Dataset):
    
    def __init__(self, wikiart_root_dir, flickr_root_dir, mode = "train", image_size = opt.size): 
        super(CycleGanDatasets, self).__init__() 
        self.image_size = image_size
        self.mode = mode
        self.wikiart_images_dir = wikiart_root_dir
        self.flickr_images_dir = flickr_root_dir
        self.wikiart_files = os.listdir(self.wikiart_images_dir)
        self.flickr_files = os.listdir(self.flickr_images_dir)
#         random.seed(0)
        random.shuffle(self.wikiart_files)
        random.shuffle(self.flickr_files)
        wikiart_length = len(self.wikiart_files)
        flickr_length = len(self.flickr_files)
        global_length = min(wikiart_length, flickr_length)
        global_step = int(0.1*global_length)
        
        if self.mode == 'train':
            self.wikiart_img_path = self.wikiart_files[0 : 7*global_step]
            self.flickr_img_path = self.flickr_files[0 : 7*global_step]
        elif self.mode == 'test':
            self.wikiart_img_path = self.wikiart_files[7*global_step : 8*global_step]
            self.flickr_img_path = self.flickr_files[7*global_step : 8*global_step]
        elif self.mode == 'val':
            self.wikiart_img_path = self.wikiart_files[8*global_step : 10*global_step]
            self.flickr_img_path = self.flickr_files[8*global_step : 10*global_step] 

    def __len__(self): 
        return len(self.wikiart_img_path)
        
    def __getitem__(self, idx): 
        wikiart_img_path = os.path.join(self.wikiart_images_dir, self.wikiart_img_path[idx])
        flickr_img_path = os.path.join(self.flickr_images_dir, self.flickr_img_path[idx])
        wikiart_img = Image.open(wikiart_img_path)
        flickr_img = Image.open(flickr_img_path)
        transform = tv.transforms.Compose([
            tv.transforms.Resize(int(1.2*self.image_size)),
            tv.transforms.RandomCrop(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        x = transform(wikiart_img)
        y = transform(flickr_img)
        return x, y


# In[ ]:


def myimshow(image, ax=plt):
    image = image.to('cpu').numpy() 
#     image = image.numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    image = (image + 1) / 2 
    image[image < 0] = 0 
    image[image > 1] = 1
    ax.figure()
    h = ax.imshow(image)
    ax.axis('off')
    return h


# In[ ]:


import time
import os
from torchvision.utils import save_image
import sys


class demo_module():
    
    def __init__(self, netG_A2B, netG_B2A, netD_A, netD_B, opt, test_loader, output_dir):
        
        loss_G = [[], []]
        loss_G_identity = [[], []]
        loss_G_GAN = [[], []]
        loss_G_cycle = [[], []]
        loss_D = [[], []]
                
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")
        
        print(checkpoint_path)
        
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        if os.path.isfile(config_path):
            self.load()
        else:
            self.save()
            
        num_epochs = self.epoch()

    
    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'netG_A2B': self.netG_A2B.state_dict(),
                'netG_B2A': self.netG_B2A.state_dict(),
                'netD_A': self.netD_A.state_dict(),
                'netD_B': self.netD_B.state_dict(),
                }

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.netG_A2B.load_state_dict(checkpoint['netG_A2B'])
        self.netG_B2A.load_state_dict(checkpoint['netG_B2A'])
        self.netD_A.load_state_dict(checkpoint['netD_A'])
        self.netD_B.load_state_dict(checkpoint['netD_B'])
        
        self.loss_G[0] = list(np.load('{}/loss_G.npy'.format(self.output_dir))[0])
        self.loss_G[1] = list(np.load('{}/loss_G.npy'.format(self.output_dir))[1])

        self.loss_G_identity[0] = list(np.load('{}/loss_G_identity.npy'.format(self.output_dir))[0])
        self.loss_G_identity[1] = list(np.load('{}/loss_G_identity.npy'.format(self.output_dir))[1])
        
        self.loss_G_GAN[0] = list(np.load('{}/loss_G_GAN.npy'.format(self.output_dir))[0])
        self.loss_G_GAN[1] = list(np.load('{}/loss_G_GAN.npy'.format(self.output_dir))[1])
        
        self.loss_G_cycle[0] = list(np.load('{}/loss_G_cycle.npy'.format(self.output_dir))[0])
        self.loss_G_cycle[1] = list(np.load('{}/loss_G_cycle.npy'.format(self.output_dir))[1])

        self.loss_D[0] = list(np.load('{}/loss_D.npy'.format(self.output_dir))[0])
        self.loss_D[1] = list(np.load('{}/loss_D.npy'.format(self.output_dir))[1])


    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        
        np.save('{}/loss_G'.format(self.output_dir), np.asarray(self.loss_G))
        np.save('{}/loss_G_identity'.format(self.output_dir), np.asarray(self.loss_G_identity))
        np.save('{}/loss_G_GAN'.format(self.output_dir), np.asarray(self.loss_G_GAN))
        np.save('{}/loss_G_cycle'.format(self.output_dir), np.asarray(self.loss_G_cycle))
        np.save('{}/loss_D'.format(self.output_dir), np.asarray(self.loss_D))
        
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.opt.cuda)
        self.load_state_dict(checkpoint)
        del checkpoint

    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.loss_G[0])

   
    def normalize(self, data):

        return (data - torch.min(data))/(torch.max(data)-torch.min(data))
      
    
    def demo(self, wikiart_subclass, flickr_subclass):
        self.netG_A2B.eval()
        self.netG_B2A.eval()

        Tensor = torch.cuda.FloatTensor if self.opt.cuda else torch.Tensor
        input_A = Tensor(self.opt.batchSize, self.opt.input_nc, self.opt.size, self.opt.size)
        input_B = Tensor(self.opt.batchSize, self.opt.output_nc, self.opt.size, self.opt.size)
        
        img_out_list = []
        
        for i, batch in enumerate(self.test_loader):#test_loader
            # Set model input
            img_list = []
            real_A = Variable(input_A.copy_(batch[0]))
            real_B = Variable(input_B.copy_(batch[1]))

            # Generate output
            fake_B = self.netG_A2B(real_A).data
            fake_A = self.netG_B2A(real_B).data

            img_out_list.append(torch.cat((self.normalize(real_A.data[0]), self.normalize(fake_B[0]),                                            self.normalize(real_B.data[0]), self.normalize(fake_A[0])),                                             dim=2).cpu().detach().numpy())    

            if i == 2:
                break
                
        for j in range(len(img_out_list)):
            img_out_list[j] = np.moveaxis(img_out_list[j], [0, 1, 2], [2, 0, 1]) 
        temp_img = np.concatenate(img_out_list, axis = 0)
        plt.figure(figsize=(70,35))
        plt.title('Style: '+wikiart_subclass +'  Content: '+flickr_subclass+'\n', fontsize=40)
        
        plt.imshow(temp_img)
        plt.axis('off')
        plt.show()

    def save_show_test_images(self):
        test_out_dir = self.output_dir + '/test'
        files_files = os.listdir(test_out_dir)
        idx_list = []
        a = 0
        for idx in range(a, a+6):
            idx_list.append(os.path.join(test_out_dir, files_files[idx]))
        img_list = []
        for item in idx_list:
            img_list.append(np.array(Image.open(item)))
        
        print(img_list[0].shape)
        
        return img_list

