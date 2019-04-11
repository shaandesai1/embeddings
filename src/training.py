
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


# In[2]:


import sys
sys.path.append('../src/')

from model import UNet
from dataset import SSSDataset
from loss import DiscriminativeLoss


# In[3]:


n_sticks = 8


# In[4]:


# Model
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
   # print('Using GPU: {} '.format(gpu_id))
    print('available')

model = UNet().to(device)


# In[5]:


# Dataset for train
train_dataset = SSSDataset(train=True, n_sticks=n_sticks)
train_dataloader = DataLoader(train_dataset, batch_size=4,
                              shuffle=False, num_workers=0, pin_memory=True)


# In[6]:


# Loss Function
criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True)
criterion_ce = nn.CrossEntropyLoss()


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
    ce_losses = []
    for batched in train_dataloader:
        images, sem_labels, ins_labels = batched
        images = images.to(device)
        sem_labels = sem_labels.to(device)
        ins_labels = ins_labels.to(device)
        model.zero_grad()

        sem_predict, ins_predict = model(images)
        loss = 0

        # Discriminative Loss
        disc_loss = criterion_disc(ins_predict,
                                   ins_labels,
                                   [n_sticks] * len(images))
        loss += disc_loss
        disc_losses.append(disc_loss.cpu().data.tolist())

        # Cross Entropy Loss
        _, sem_labels_ce = sem_labels.max(1)
        ce_loss = criterion_ce(sem_predict.permute(0, 2, 3, 1)                                   .contiguous().view(-1, 2),
                               sem_labels_ce.view(-1))
        loss += ce_loss
        ce_losses.append(ce_loss.cpu().data.tolist())

        loss.backward()
        optimizer.step()
    disc_loss = np.mean(disc_losses)
    ce_loss = np.mean(ce_losses)
    #print(f'DiscriminativeLoss: {disc_loss:.4f}')
    #print(f'CrossEntropyLoss: {ce_loss:.4f}')
    scheduler.step(disc_loss)
    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        modelname = 'model.pth'
        torch.save(model.state_dict(), modelname)

