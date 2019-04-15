import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os

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
    
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.coco = COCO(annFile)
        self.root = root
        self.ids = list(self.coco.imgs.keys())
    
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
        
        #print(anns[0]['category_id'])
        #cats = coco.loadCats(coco.getCatIds())
        #nms=[cat['id'] for cat in cats]
        #print(list(self.coco.cats.keys()))
        
        
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
        for i in range(len(anns)):
            arr_img = coco.annToMask(anns[i])
            resized = cv2.resize(arr_img,dsize=(shap[1],shap[0]),interpolation=cv2.INTER_NEAREST)
            if np.max(resized) != 1:
                resized[np.nonzero(resized)] = 1
            bgnd.append(resized)
            target.append(resized)
        bgnd = sum(bgnd)
        bgnd[np.nonzero(bgnd)] = 1
        bgnd = 1 - bgnd
        target.append(bgnd)
        
        
        #padding
        pad_amt = new_size - min_size
        pad_left = pad_amt//2
        pad_right = pad_amt - pad_left
        if sz[0] > sz[1]:
            padding = (0,pad_left,0,pad_right)
        else:
            padding = (pad_left,0,pad_right,0)
        
        img = ImageOps.expand(img, padding)
        
        for i in range(len(anns)+1):
            if sz[0] > sz[1]:
                target[i] = cv2.copyMakeBorder(target[i],0,0,pad_left,pad_right,cv2.BORDER_CONSTANT,value=0)
            else:
                target[i] = cv2.copyMakeBorder(target[i],pad_left,pad_right,0,0,cv2.BORDER_CONSTANT,value=0)
        
        target = np.array(target)
        
        
        new_target = np.zeros((92,256,256))
        
        for i in range(len(anns)):
            #new_target = new_target + target[i,:,:]*anns[i]['category_id']
            new_target[anns[i]['category_id'],:,:] = new_target[anns[i]['category_id'],:,:] + target[i,:,:]
        new_target[0,:,:] = new_target[0,:,:] + target[i+1,:,:]
        target = torch.from_numpy(new_target)
        
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    
    def __len__(self):
        return len(self.ids)
