
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
from new_loss import DiscriminativeLoss
import torchvision.models as models
from torch.nn import DataParallel


import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from sklearn.metrics import jaccard_score as jacc

from pycocotools.coco import COCO

global_classes = 40
#global_classes = global_classes
rejects = []
annFile='/data/shaan/annotations/instances_train2017.json'
coco=COCO(annFile)
#find the top 'globalclasses' categories and their ids
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
idset = []
catcount = []

for i,name in enumerate(nms):
    catIds = coco.getCatIds(catNms=name)
    imgIds = coco.getImgIds(catIds=catIds)
    catcount.append(len(imgIds))

indices = np.flip(np.argsort(catcount)[-global_classes:])



topk_catnames = []
for i in range(global_classes):
    topk_catnames.append(nms[indices[i]])
    







#if pretrain == 1:


vgg13 = models.vgg13(pretrained=True)
model = UNet()

dctvgg = vgg13.state_dict()
dct = model.state_dict()

dct['inc.conv.conv.0.weight'].data.copy_(dctvgg['features.0.weight'])#
dct['inc.conv.conv.0.bias'].data.copy_(dctvgg['features.0.bias'])
#
dct['inc.conv.conv.3.weight'].data.copy_(dctvgg['features.2.weight'])
dct['inc.conv.conv.3.bias'].data.copy_(dctvgg['features.2.bias'])
#
dct['down1.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.5.weight'])
dct['down1.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.5.bias'])
#
dct['down1.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.7.weight'])
dct['down1.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.7.bias'])
#
dct['down2.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.10.weight'])
dct['down2.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.10.bias'])
#
dct['down2.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.12.weight'])
dct['down2.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.12.bias'])
#
dct['down3.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.15.weight'])
dct['down3.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.15.bias'])
#
dct['down3.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.17.weight'])
dct['down3.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.17.bias'])
#
dct['down4.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.20.weight'])
dct['down4.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.20.bias'])
#
dct['down4.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.22.weight'])
dct['down4.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.22.bias'])
#
model.load_state_dict(dct)


writer = SummaryWriter()


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

# intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    
    return overlaps

def get_bin_map(true_msk,pred_msk):
    ids = list(set(np.unique(true_msk)) -set([0,99]))
    all_collate = np.zeros((256,256,len(ids),2))
    #     for i,val in enumerate(ids):
    #         tsmsk = (true_msk == val)*1
    #         pmsk = (pred_msk == val)*1
    #         all_collate[:,:,i,0] = tsmsk
    #         all_collate[:,:,i,1] = pmsk
    all_collate[:,:,:,0] = true_msk.unsqueeze(2).expand(256,256,len(ids)).float() == (torch.ones((256,256,len(ids)))*torch.Tensor(ids)).float()
    all_collate[:,:,:,1] = pred_msk.unsqueeze(2).expand(256,256,len(ids)).float() == (torch.ones((256,256,len(ids)))*torch.Tensor(ids)).float()
    
    
    return compute_overlaps_masks(all_collate[:,:,:,1],all_collate[:,:,:,0])

# Model
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('using gpus')
    model = DataParallel(model,device_ids=range(torch.cuda.device_count()))
#model.load_state_dict(torch.load('model.pth',map_location='cpu'))
model.to(device)


def my_collate(batch):
    img = [item[0] for item in batch]  # just form a list of tensor
    
    mask = [item[1] for item in batch]
    
    instance = [item[2] for item in batch]
    
    annid = [item[3] for item in batch]

    coord = [item[4] for item in batch]
    #instance = torch.LongTensor(instance)
    #annid = torch.LongTensor(annid)
    return [img,mask,instance,annid,coord]

#coco dataset training
train_df = CocoDetection('/data/shaan/train2017','/data/shaan/annotations/instances_train2017.json',catnames=topk_catnames,transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
train_dataloader = DataLoader(train_df, batch_size =16, shuffle = True, num_workers = 2,collate_fn = my_collate)
val_df = CocoDetection('/data/shaan/val2017','/data/shaan/annotations/instances_val2017.json',catnames=topk_catnames,transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
val_dataloader = DataLoader(val_df, batch_size =16, num_workers = 2,collate_fn=my_collate)



data_dict = {'train': train_dataloader, 'validation': val_dataloader}
# Loss Function
#criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                    delta_dist=1.5,
#                                    norm=2,
#                                    usegpu=True)

weights = [1,1,3.7,3.9,1,8.5,12.6/2,3.3,22.1/2,4.1,7.1,24.1/2,10.7/2,23.6/2,14.3/2,19.1/2,7.3,10.2/2,4.2,5.4,51.4/2,1592/40,38.1/2,4.0,8.3,3.3,31.6/2,2.5,14.0/2,3.3,67.9/2,74.5/2,6.6,5.1,21.2/2,40.1/2,42.2/2,8.1,15.2/2,14.8/2,3.0]
#e padding
criterion_ce = nn.CrossEntropyLoss(ignore_index=99,weight=torch.Tensor(weights)).cuda()
discriminative_loss = DiscriminativeLoss().cuda()
# Optimizer
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=1e-3)
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
            semantic_ious = 0.
            for batched in data_dict[phase]:
                print('batch')
                images, sem_labels,instances,annid,coords = batched
                images = torch.stack(images)
                coords = torch.stack(coords)
                sem_labels = torch.stack(sem_labels)
                images = images.float().to(device)
                coords = coords.float().to(device)
                sem_labels = sem_labels.long().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                
                    inst_predict,sem_predict = model(images)
                    inst_predict = torch.cat([inst_predict,coords],dim=1)
                    ce_loss = criterion_ce(sem_predict,sem_labels)
                    disc_loss =0.1*discriminative_loss(inst_predict,instances,annid,epoch)
                    
                    ss = F.softmax(sem_predict,dim=1)
                    yp = torch.argmax(ss,dim=1).cpu()
                    yt = sem_labels.cpu()
                    
                    loss = ce_loss + disc_loss
   
                    jacc_bvalue = jacc(yt.numpy().reshape(-1),yp.numpy().reshape(-1),average='weighted')
                
                    if phase == 'train':
                        n_iter_tr += 1
                        
 
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('jacc(iou)_train_batch',jacc_bvalue,n_iter_tr)
                        writer.add_scalar('CELoss_train_batch',loss,n_iter_tr)
                    else:
                        n_iter_val += 1
                        for i in range(images.size(0)):
                            overlap = np.mean(np.diag(get_bin_map(yt[i,:,:],yp[i,:,:].float()*torch.Tensor(np.where(yt[i,:,:]!=99,1.,0.)))))
                            semantic_ious += overlap
                            writer.add_scalar('semiouvalbatch',overlap,n_iter_val)
                        writer.add_scalar('jacc(iou)_val_batch',jacc_bvalue,n_iter_val)
                        writer.add_scalar('CELoss_val_batch',loss,n_iter_val)
                    print(loss)
                    running_losses += loss.cpu().data.tolist()[0]*images.size(0)
                    running_ious += jacc_bvalue*images.size(0)
            avg_loss =running_losses/len(data_dict[phase].dataset)
            avg_iou = running_ious/len(data_dict[phase].dataset)
            
            if phase == 'train':
                writer.add_scalar('jacc(iou)_train_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_train_epoch',avg_loss,epoch)
            else:
                writer.add_scalar('jacc(iou)_val_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_val_epoch',avg_loss,epoch)
                writer.add_scalar('sem val epoch',semantic_ious/len(data_dict[phase].dataset),epoch)
         #   scheduler.step(loss)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    print('Best Model!')
                    modelname = 'model.pth'
                    torch.save(model.state_dict(), modelname)



train_model(model,optimizer,scheduler,num_epochs=30)
