import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os

from PIL import Image, ImageOps
import PIL
from skimage.transform import resize
from pycocotools.coco import COCO

class SSSDataset(Dataset):
    def __init__(self, train, n_sticks=8, data_size=512):
        super().__init__()
        self.train = train
        self.n_sticks = n_sticks
        self.data_size = data_size
        self.height = 256
        self.width = 256

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        while True:
            img = np.ones((self.height, self.width), dtype=np.uint8) * 255
            ins = np.zeros((0, self.height, self.width), dtype=np.uint8)
            for _ in range(self.n_sticks):
                x = np.random.randint(30, 225)
                y = np.random.randint(30, 225)
                w = 15
                h = np.random.randint(80, 100)
                theta = np.random.randint(-90, 90)
                rect = ([x, y], [w, h], theta)
                box = np.int0(cv2.boxPoints(rect))

                gt = np.zeros_like(img)
                gt = cv2.fillPoly(gt, [box], 1)
                ins[:, gt != 0] = 0
                ins = np.concatenate([ins, gt[np.newaxis]])
                img = cv2.fillPoly(img, [box], 255)
                img = cv2.drawContours(img, [box], 0, 0, 2)

            # minimum area of stick
            if np.sum(np.sum(ins, axis=(1, 2)) < 400) == 0:
                break

        if self.train:
            sem = np.zeros_like(img, dtype=bool)
            sem[np.sum(ins, axis=0) != 0] = True
            sem = np.stack([~sem, sem]).astype(np.uint8)

            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            # 2 * height * width
            sem = torch.Tensor(sem)
            # n_sticks * height * width
            ins = torch.Tensor(ins)
            return img, sem, ins
        else:
            # 1 * height * width
            img = torch.Tensor(img[np.newaxis])
            return img

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
        
        Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
        """
    
    def __init__(self, root, annFile, catnames=None,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.coco = COCO(annFile)
        self.root = root
        ctr = 0
        global_classes = len(catnames)
        self.global_classes = global_classes
        rejects = []

        idset = []
        for i,name in enumerate(catnames):
            catIds = self.coco.getCatIds(catNms=name)
            imgIds = self.coco.getImgIds(catIds=catIds)
            idset.append(imgIds)

        init_ids = idset[0]

        for i in range(1,global_classes):
            init_ids = list(set().union(init_ids,idset[i]))

        #self.ids = list(set(self.ids).intersection(init_ids))
        for i,val in enumerate(init_ids):
            annIds = self.coco.getAnnIds(imgIds=val)
            anns = self.coco.loadAnns(annIds)
            if anns == []:
                rejects.append(val)
                ctr+=1 
           # else:
           #     for i in range(len(anns)):
           #         if np.sum(self.coco.annToMask(anns[i])) < 5:
           #             rejects.append(val)
           #             break

        self.ids = list(set(init_ids) - set(rejects))

        dct = {}

        for i in range(global_classes):
            #print(nms[indices[i]])
            val_tmp = self.coco.getCatIds(catNms=catnames[i])
            #print(val_tmp)
            dct[val_tmp[0]] = 1 + i 
        
        self.reindex = dct

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            
            Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
            """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        sz = img.size
        new_size = 256
        
        
        # resizing the image on long dim, maintain aspect ratio
        if sz[0] > sz[1]:
            ratio = new_size/sz[0]
            min_size = int(ratio*sz[1])
            shap = (new_size,min_size)
            img = img.resize(shap,resample = PIL.Image.BILINEAR)
        else:
            ratio = new_size/sz[1]
            min_size = int(ratio*sz[0])
            shap = (min_size,new_size)
            img = img.resize(shap,resample = PIL.Image.BILINEAR)
        
        #resizing the target masks + adding background mask
        target = []
        bgnd = []
        new_anns = []
        
        
        instances = []
        
        
        
        for i in range(len(anns)):
            if anns[i]['category_id'] in self.reindex:
                arr_img = coco.annToMask(anns[i])
                resized = cv2.resize(arr_img,dsize=(shap[0],shap[1]),interpolation=cv2.INTER_NEAREST)
                if np.max(resized) != 1:
                    resized[np.nonzero(resized)] = 1
                        #print(np.max(resized))
                if np.sum(resized) >= 1:
                    bgnd.append(resized)
                    new_anns.append(self.reindex[anns[i]['category_id']])
                    instances.append(resized)
                    target.append(resized*new_anns[-1])
                
        #bgnd = sum(bgnd)
        #bgnd[np.nonzero(bgnd)] = 1
        #bgnd = 1 - bgnd
        #target.append(bgnd)
        
        sort_order = np.argsort(new_anns)
        sorted_anns = np.sort(new_anns)
        instances = np.array(instances)
        #print(instances.shape)
        if sort_order.size == 0:
            instances = np.zeros((1,shap[1],shap[0]),dtype='uint8')
            #sort_order = [0]
            sorted_anns = np.ones(1,dtype='int64')
            target = np.zeros((shap[1],shap[0]),dtype='uint8') 
        else:
            instances = instances[sort_order,:,:]
            target = np.array(target)
            target = np.max(target,axis=0)
        #padding
        pad_amt = new_size - min_size
        pad_left = pad_amt//2
        pad_right = pad_amt - pad_left
        if sz[0] > sz[1]:
            padding = (0,pad_left,0,pad_right)
        else:
            padding = (pad_left,0,pad_right,0)
        
        img = ImageOps.expand(img, padding)
        pad_instances = np.zeros((len(sorted_anns),256,256))
        for i in range(len(sorted_anns)):
            if sz[0] > sz[1]:
                pad_instances[i,:,:] = cv2.copyMakeBorder(instances[i,:,:],pad_left,pad_right,0,0,cv2.BORDER_CONSTANT,value=0)
            else:
                pad_instances[i,:,:] = cv2.copyMakeBorder(instances[i,:,:],0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)

        ones = torch.ones(shap[1],shap[0])
        eyes = torch.arange(-1,1,2/shap[0]) * ones
        jays = (torch.arange(-1,1,2/shap[1]).t() * ones).t()


        #for i in range(len(target)):
        if sz[0] > sz[1]:
            target = cv2.copyMakeBorder(target,pad_left,pad_right,0,0,cv2.BORDER_CONSTANT,value=99)
            eyes = cv2.copyMakeBorder(eyes,pad_left,pad_right,0,0,cv2.BORDER_CONSTANT,value=0)
            jays = cv2.copyMakeBorder(jays,pad_left,pad_right,0,0,cv2.BORDER_CONSTANT,value=0)
        else:
            target = cv2.copyMakeBorder(target,0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=99)
            eyes = cv2.copyMakeBorder(eyes,0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)
            jays = cv2.copyMakeBorder(jays,0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)
        
        coords = torch.stack([eyes,jays])
        
        target = torch.from_numpy(target)
        instances = torch.from_numpy(pad_instances)
        #sorted_anns = torch.from_numpy(sorted_anns)      
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            pass
        return img, target, instances, sorted_anns,coords
    
    
    def __len__(self):
        return len(self.ids)
