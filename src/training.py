
# coding: utf-8

import numpy as np
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
sys.path.append('../src/')

from model import UNet
from dataset import SSSDataset
from dataset import CocoDetection
from loss import DiscriminativeLoss


from torch.nn.parallel import DataParallel

#parallel_model = DataParallelModel(model)             # Encapsulate the model
#parallel_loss  = DataParallelCriterion(loss_function) # Encapsulate the loss function


# Model
#gpu_id = 0
#device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
#if torch.cuda.is_available():
   # print('Using GPU: {} '.format(gpu_id))
#    print('available')




model = DataParallel(UNet())
#model = UNet().to(device)

# Dataset for train with sticks
#train_dataset = SSSDataset(train=True, n_sticks=n_sticks)
#train_dataloader = DataLoader(train_dataset, batch_size=4,
#                              shuffle=False, num_workers=0, pin_memory=True)

#coco dataset training

train_df = CocoDetection('/data/shaan/train2017','/data/shaan/annotations/instances_train2017.json',transform = transforms.ToTensor(),target_transform=transforms.ToTensor())

train_dataloader = DataLoader(train_df, batch_size = 32, shuffle = True, num_workers = 2)

#val_df = CocoDetection('/data/shaan/val2017','/data/shaan/annotations/instances_val2017.json',transform = transforms.toTensor(),target_transform=transforms.toTensor())

#val_df = CocoDetection('/data/shaan/test2017','/data/shaan/annotations/instances_test2017.json',transform = transforms.toTensor(),target_transform=transforms.toTensor())



# Loss Function
criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True)
#criterion_ce = nn.CrossEntropyLoss()


# In[7]:


# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.1,
                                                 patience=10,
                                                 verbose=True)


# In[24]:


# Train
model_dir = 'model'

best_loss = np.inf
for epoch in range(300):
    #print(f'epoch : {epoch}')
    disc_losses = []
    #ce_losses = []
    for batched in train_dataloader:
        images, ins_labels = batched
 #       images = images.to(device)
 #       ins_labels = ins_labels.to(device)
        images = images.float()
        ins_labels = ins_labels.float()
        model.zero_grad()

        ins_predict = model(images)
        loss = 0

        
        
        # Discriminative Loss
        disc_loss = criterion_disc(ins_predict,
                                   ins_labels,
                                   [92] * len(images))
        loss += disc_loss
        disc_losses.append(disc_loss.cpu().data.tolist())

    
        loss.backward()
        optimizer.step()

    disc_loss = np.mean(disc_losses)
    #print(f'DiscriminativeLoss: {disc_loss:.4f}')
    #print(f'CrossEntropyLoss: {ce_loss:.4f}')
    scheduler.step(disc_loss)
    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        modelname = 'model.pth'
        torch.save(model.state_dict(), modelname)

