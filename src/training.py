
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
train_dataloader = DataLoader(train_df, batch_size =32, shuffle = True, num_workers = 2)
val_df = CocoDetection('/data/shaan/val2017','/data/shaan/annotations/instances_val2017.json',transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
val_dataloader = DataLoader(val_df, batch_size =32, num_workers = 2)


data_dict = {'train': train_dataloader, 'validation': val_dataloader}
# Loss Function
#criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                    delta_dist=1.5,
#                                    norm=2,
#                                    usegpu=True)

#ignore padding
criterion_ce = nn.CrossEntropyLoss(ignore_index=99).cuda()

# Optimizer
parameters = model.parameters()
optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                             mode='min',
    #                                             factor=0.1,
    #                                             patience=10,
#                                             verbose=True)

scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)


def train_model(model,optimizer,scheduler,num_epochs=10):
    #early = time.time()
    # Train
    n_iter_tr = 0
    n_iter_val = 0
    best_iou = -np.inf
    for epoch in range(num_epochs):
        #print(f'epoch : {epoch}')
        
        print('epoch')
        
        for phase in ['train','validation']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()
            
            running_losses = 0.
            running_ious = 0.
           
            for batched in data_dict[phase]:
                print('batch')
                images, ins_labels = batched
                images = images.float().to(device)
                ins_labels = ins_labels.long().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                
                    ins_predict = model(images)
                    loss = criterion_ce(ins_predict,ins_labels)
                    
                    
                    ss = F.softmax(ins_predict,dim=1)
                    yp = torch.argmax(ss,dim=1).cpu().numpy().reshape(-1)
                    yt = ins_labels.cpu().numpy().reshape(-1)
                
                    jacc_bvalue = jacc(yt,yp)
                
                    if phase == 'train':
                        n_iter_tr += 1
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('jacc(iou)_train_batch',jacc_bvalue,n_iter_tr)
                        writer.add_scalar('CELoss_train_batch',loss,n_iter_tr)
                    else:
                        n_iter_val += 1
                        writer.add_scalar('jacc(iou)_val_batch',jacc_bvalue,n_iter_val)
                        writer.add_scalar('CELoss_val_batch',loss,n_iter_val)

                    running_losses += loss.cpu().data.tolist()*images.size(0)
                    running_ious += jacc_bvalue*images.size(0)
            avg_loss =running_losses/len(data_dict[phase].dataset)
            avg_iou = running_ious/len(data_dict[phase].dataset)
            
            if phase == 'train':
                writer.add_scalar('jacc(iou)_train_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_train_epoch',avg_loss,epoch)
            else:
                writer.add_scalar('jacc(iou)_val_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_val_epoch',avg_loss,epoch)
         #   scheduler.step(loss)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    print('Best Model!')
                    modelname = 'model.pth'
                    torch.save(model.state_dict(), modelname)



train_model(model,optimizer,scheduler,num_epochs=30)
