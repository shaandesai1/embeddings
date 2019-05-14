
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
from sklearn.metrics import jaccard_similarity_score as jacc

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


vgg16= models.vgg16_bn(pretrained=True)
model = UNet()

dctvgg = vgg16.state_dict()
dct = model.state_dict()

dct['inc.conv.conv.0.weight'].data.copy_(dctvgg['features.0.weight'])#
dct['inc.conv.conv.0.bias'].data.copy_(dctvgg['features.0.bias'])
dct['inc.conv.conv.1.weight'].data.copy_(dctvgg['features.1.weight'])#
dct['inc.conv.conv.1.bias'].data.copy_(dctvgg['features.1.bias'])
dct['inc.conv.conv.1.running_mean'].data.copy_(dctvgg['features.1.running_mean'])#
dct['inc.conv.conv.1.running_var'].data.copy_(dctvgg['features.1.running_var'])


#
dct['inc.conv.conv.3.weight'].data.copy_(dctvgg['features.3.weight'])
dct['inc.conv.conv.3.bias'].data.copy_(dctvgg['features.3.bias'])
dct['inc.conv.conv.4.weight'].data.copy_(dctvgg['features.4.weight'])#
dct['inc.conv.conv.4.bias'].data.copy_(dctvgg['features.4.bias'])
dct['inc.conv.conv.4.running_mean'].data.copy_(dctvgg['features.4.running_mean'])#
dct['inc.conv.conv.4.running_var'].data.copy_(dctvgg['features.4.running_var'])


#
dct['down1.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.7.weight'])
dct['down1.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.7.bias'])
dct['down1.mpconv.1.conv.1.weight'].data.copy_(dctvgg['features.8.weight'])#
dct['down1.mpconv.1.conv.1.bias'].data.copy_(dctvgg['features.8.bias'])
dct['down1.mpconv.1.conv.1.running_mean'].data.copy_(dctvgg['features.8.running_mean'])#
dct['down1.mpconv.1.conv.1.running_var'].data.copy_(dctvgg['features.8.running_var'])

dct['down1.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.10.weight'])
dct['down1.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.10.bias'])
dct['down1.mpconv.1.conv.4.weight'].data.copy_(dctvgg['features.11.weight'])#
dct['down1.mpconv.1.conv.4.bias'].data.copy_(dctvgg['features.11.bias'])
dct['down1.mpconv.1.conv.4.running_mean'].data.copy_(dctvgg['features.11.running_mean'])#
dct['down1.mpconv.1.conv.4.running_var'].data.copy_(dctvgg['features.11.running_var'])


dct['down2.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.14.weight'])
dct['down2.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.14.bias'])
dct['down2.mpconv.1.conv.1.weight'].data.copy_(dctvgg['features.15.weight'])#
dct['down2.mpconv.1.conv.1.bias'].data.copy_(dctvgg['features.15.bias'])
dct['down2.mpconv.1.conv.1.running_mean'].data.copy_(dctvgg['features.15.running_mean'])#
dct['down2.mpconv.1.conv.1.running_var'].data.copy_(dctvgg['features.15.running_var'])

dct['down2.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.17.weight'])
dct['down2.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.17.bias'])
dct['down2.mpconv.1.conv.4.weight'].data.copy_(dctvgg['features.18.weight'])#
dct['down2.mpconv.1.conv.4.bias'].data.copy_(dctvgg['features.18.bias'])
dct['down2.mpconv.1.conv.4.running_mean'].data.copy_(dctvgg['features.18.running_mean'])#
dct['down2.mpconv.1.conv.4.running_var'].data.copy_(dctvgg['features.18.running_var'])

dct['down2.mpconv.1.conv.6.weight'].data.copy_(dctvgg['features.20.weight'])
dct['down2.mpconv.1.conv.6.bias'].data.copy_(dctvgg['features.20.bias'])
dct['down2.mpconv.1.conv.7.weight'].data.copy_(dctvgg['features.21.weight'])#
dct['down2.mpconv.1.conv.7.bias'].data.copy_(dctvgg['features.21.bias'])
dct['down2.mpconv.1.conv.7.running_mean'].data.copy_(dctvgg['features.21.running_mean'])#
dct['down2.mpconv.1.conv.7.running_var'].data.copy_(dctvgg['features.21.running_var'])


dct['down3.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.24.weight'])
dct['down3.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.24.bias'])
dct['down3.mpconv.1.conv.1.weight'].data.copy_(dctvgg['features.25.weight'])#
dct['down3.mpconv.1.conv.1.bias'].data.copy_(dctvgg['features.25.bias'])
dct['down3.mpconv.1.conv.1.running_mean'].data.copy_(dctvgg['features.25.running_mean'])#
dct['down3.mpconv.1.conv.1.running_var'].data.copy_(dctvgg['features.25.running_var'])

dct['down3.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.27.weight'])
dct['down3.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.27.bias'])
dct['down3.mpconv.1.conv.4.weight'].data.copy_(dctvgg['features.28.weight'])#
dct['down3.mpconv.1.conv.4.bias'].data.copy_(dctvgg['features.28.bias'])
dct['down3.mpconv.1.conv.4.running_mean'].data.copy_(dctvgg['features.28.running_mean'])#
dct['down3.mpconv.1.conv.4.running_var'].data.copy_(dctvgg['features.28.running_var'])

dct['down3.mpconv.1.conv.6.weight'].data.copy_(dctvgg['features.30.weight'])
dct['down3.mpconv.1.conv.6.bias'].data.copy_(dctvgg['features.30.bias'])
dct['down3.mpconv.1.conv.7.weight'].data.copy_(dctvgg['features.31.weight'])#
dct['down3.mpconv.1.conv.7.bias'].data.copy_(dctvgg['features.31.bias'])
dct['down3.mpconv.1.conv.7.running_mean'].data.copy_(dctvgg['features.31.running_mean'])#
dct['down3.mpconv.1.conv.7.running_var'].data.copy_(dctvgg['features.31.running_var'])


dct['down4.mpconv.1.conv.0.weight'].data.copy_(dctvgg['features.34.weight'])
dct['down4.mpconv.1.conv.0.bias'].data.copy_(dctvgg['features.34.bias'])
dct['down4.mpconv.1.conv.1.weight'].data.copy_(dctvgg['features.35.weight'])#
dct['down4.mpconv.1.conv.1.bias'].data.copy_(dctvgg['features.35.bias'])
dct['down4.mpconv.1.conv.1.running_mean'].data.copy_(dctvgg['features.35.running_mean'])#
dct['down4.mpconv.1.conv.1.running_var'].data.copy_(dctvgg['features.35.running_var'])

dct['down4.mpconv.1.conv.3.weight'].data.copy_(dctvgg['features.37.weight'])
dct['down4.mpconv.1.conv.3.bias'].data.copy_(dctvgg['features.37.bias'])
dct['down4.mpconv.1.conv.4.weight'].data.copy_(dctvgg['features.38.weight'])#
dct['down4.mpconv.1.conv.4.bias'].data.copy_(dctvgg['features.38.bias'])
dct['down4.mpconv.1.conv.4.running_mean'].data.copy_(dctvgg['features.38.running_mean'])#
dct['down4.mpconv.1.conv.4.running_var'].data.copy_(dctvgg['features.38.running_var'])

dct['down4.mpconv.1.conv.6.weight'].data.copy_(dctvgg['features.40.weight'])
dct['down4.mpconv.1.conv.6.bias'].data.copy_(dctvgg['features.40.bias'])
dct['down4.mpconv.1.conv.7.weight'].data.copy_(dctvgg['features.41.weight'])#
dct['down4.mpconv.1.conv.7.bias'].data.copy_(dctvgg['features.41.bias'])
dct['down4.mpconv.1.conv.7.running_mean'].data.copy_(dctvgg['features.41.running_mean'])#
dct['down4.mpconv.1.conv.7.running_var'].data.copy_(dctvgg['features.41.running_var']);






#
model.load_state_dict(dct)


writer = SummaryWriter()


# Model
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print('using gpus')
    model = DataParallel(model,device_ids=range(torch.cuda.device_count()))
#model.load_state_dict(torch.load('model.pth',map_location='cpu'))
model.to(device)


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
    overlaps = intersections / (union+1e-6)
    
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





def my_collate(batch):
    img = [item[0] for item in batch]  # just form a list of tensor
    
    mask = [item[1] for item in batch]
    
    instance = [item[2] for item in batch]
    
    annid = [item[3] for item in batch]
    #instance = torch.LongTensor(instance)
    #annid = torch.LongTensor(annid)
    return [img,mask,instance,annid]

#coco dataset training
train_df = CocoDetection('/data/shaan/train2017','/data/shaan/annotations/instances_train2017.json',catnames=topk_catnames,transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
train_dataloader = DataLoader(train_df, batch_size =15, shuffle = True, num_workers = 2,collate_fn = my_collate)
val_df = CocoDetection('/data/shaan/val2017','/data/shaan/annotations/instances_val2017.json',catnames=topk_catnames,transform = transforms.ToTensor(),target_transform=transforms.ToTensor())
val_dataloader = DataLoader(val_df, batch_size =15, num_workers = 2,collate_fn=my_collate)



data_dict = {'train': train_dataloader, 'validation': val_dataloader}
# Loss Function
#criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                    delta_dist=1.5,
#                                    norm=2,
#                                    usegpu=True)
weight = [1,3,37,39,7,85,126,33,221,41,71,241,107,236,143,191,73,102,42,54,514,1592,381,40,83,33,316,25,140,33,679,745,66,51,212,401,422,81,152,148,30]
#ignore padding
criterion_ce = nn.CrossEntropyLoss(ignore_index=99,weight=torch.Tensor(weight)).cuda()
discriminative_loss = DiscriminativeLoss().cuda()
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
            semantic_ious = 0.
            for batched in data_dict[phase]:
                print('batch')
                images, sem_labels,instances,annid = batched
                images = torch.stack(images)
                sem_labels = torch.stack(sem_labels)
                images = images.float().to(device)
                sem_labels = sem_labels.long().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase =='train'):
                
                    inst_predict,sem_predict = model(images)
                    ce_loss = criterion_ce(sem_predict,sem_labels)
                    disc_loss = 0.1*discriminative_loss(inst_predict,instances,annid)
                    
                    ss = F.softmax(sem_predict,dim=1)
                    yp = torch.argmax(ss,dim=1).cpu()
                    yt = sem_labels.cpu()
                                
                    loss = ce_loss + disc_loss
                    jacc_bvalue = jacc(yt.numpy().reshape(-1),yp.numpy().reshape(-1))
                
                    if phase == 'train':
                        n_iter_tr += 1
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar('jacc(iou)_train_batch',jacc_bvalue,n_iter_tr)
                        writer.add_scalar('CELoss_train_batch',loss,n_iter_tr)
                    else:
                        n_iter_val += 1
                        for i in range(images.size(0)):
                            overlap = np.mean(np.diag(get_bin_map(yt[i,:,:],yp[i,:,:])))
                            semantic_ious += overlap
                            print("overlap:" +str(overlap))
                            writer.add_scalar('semantic(iou)_val_batch',overlap,n_iter_val)
                        
                        writer.add_scalar('jacc(iou)_val_batch',jacc_bvalue,n_iter_val)
                        writer.add_scalar('CELoss_val_batch',loss,n_iter_val)
                    print(loss,ce_loss,disc_loss)
                    running_losses += loss.cpu().data.tolist()[0]*images.size(0)
                    running_ious += jacc_bvalue*images.size(0)
            avg_loss =running_losses/len(data_dict[phase].dataset)
            avg_iou = running_ious/len(data_dict[phase].dataset)
           
            if phase == 'train':
                writer.add_scalar('jacc(iou)_train_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_train_epoch',avg_loss,epoch)
            else:
                writer.add_scalar('semantic_iou_avg_epoch)',semantic_ious/len(data_dict[phase].dataset),epoch)
                writer.add_scalar('jacc(iou)_val_epoch',avg_iou,epoch)
                writer.add_scalar('CELoss_val_epoch',avg_loss,epoch)
         #   scheduler.step(loss)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    print('Best Model!')
                    modelname = 'model.pth'
                    torch.save(model.state_dict(), modelname)



train_model(model,optimizer,scheduler,num_epochs=30)
