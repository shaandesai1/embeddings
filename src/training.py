
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
import torchvision.models as models
from torch.nn import DataParallel


import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from sklearn.metrics import jaccard_similarity_score as jacc



vgg13 = models.vgg13(pretrained=True)
model = UNet()

dctvgg = vgg13.state_dict()
dct = model.state_dict()

dct['inc.conv.conv.0.weight'].data.copy_(dctvgg['features.0.weight'])
dct['inc.conv.conv.0.bias'].data.copy_(dctvgg['features.0.bias'])

dct['inc.conv.conv.3.weight'].data.copy_(dctvgg['features.2.weight'])
dct['inc.conv.conv.3.bias'].data.copy_(dctvgg['features.2.bias'])

dct['down1.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.5.weight'])
dct['down1.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.5.bias'])

dct['down1.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.7.weight'])
dct['down1.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.7.bias'])

dct['down2.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.10.weight'])
dct['down2.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.10.bias'])

dct['down2.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.12.weight'])
dct['down2.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.12.bias'])

dct['down3.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.15.weight'])
dct['down3.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.15.bias'])

dct['down3.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.17.weight'])
dct['down3.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.17.bias'])

dct['down4.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.20.weight'])
dct['down4.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.20.bias'])

dct['down4.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.22.weight'])
dct['down4.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.22.bias'])

model.load_state_dict(dct)


writer = SummaryWriter()


# Model
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('using gpus')
    model = DataParallel(model,device_ids=range(torch.cuda.device_count()))

model.to(device)

#coco dataset training
train_df = CocoDetection('/data/shaan/train2017','/data/shaan/annotations/instances_train2017.json',transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
train_dataloader = DataLoader(train_df, batch_size =50, shuffle = True, num_workers = 2)


# Loss Function
#criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                    delta_dist=1.5,
#                                    norm=2,
#                                    usegpu=True)
criterion_ce = nn.CrossEntropyLoss(ignore_index=99).cuda()

# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min',
                                                 factor=0.1,
                                                 patience=10,
                                                 verbose=True)


# Train
model_dir = 'model'
n_iter = 0
best_loss = np.inf
for epoch in range(15):
    #print(f'epoch : {epoch}')
    disc_losses = []
    print('epoch')
    
    for batched in train_dataloader:
        n_iter += 1
        print('batch')
        images, ins_labels = batched
        images = images.float().to(device)
        ins_labels = ins_labels.long().to(device)
        model.zero_grad()

        ins_predict = model(images)
        loss = 0
        ss = F.softmax(ins_predict,dim=1)
        yp = torch.argmax(ss,dim=1).numpy().reshape(-1)
        yt = ins_labels.numpy().reshape(-1)
        
        # Discriminative Loss
        #disc_loss = criterion_disc(ins_predict,
        #                           ins_labels,
        #                           [41] * len(images))
        #loss += disc_loss
        loss = criterion_ce(ins_predict,ins_labels)
        disc_losses.append(loss.cpu().data.tolist())
        
        loss.backward()
        optimizer.step()
        writer.add_scalar('jacc(iou)',jacc(yt,yp),n_iter)
        writer.add_scalar('scalar1',loss,n_iter)
    disc_loss = np.mean(disc_losses)
    scheduler.step(loss)
    if disc_loss < best_loss:
        best_loss = disc_loss
        print('Best Model!')
        modelname = 'model.pth'
        torch.save(model.state_dict(), modelname)

