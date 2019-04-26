"""
    This is the implementation of following paper:
    https://arxiv.org/pdf/1802.05591.pdf
    This implementation is based on following code:
    https://github.com/Wizaron/instance-segmentation-pytorch
    """
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np


class DiscriminativeLoss(_Loss):
    
    def __init__(self,eps = 1e-6,delta_dist_intra = 1.1, delta_dist_inter=2.,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001, delta_var = 0.5,
                 usegpu=True, size_average=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.eps = eps
        self.delta_dist_intra = delta_dist_intra
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_dist_inter = delta_dist_inter
        self.delta_var = delta_var
        self.norm = norm
        self.usegpu = usegpu
        assert self.norm in [1, 2]
    
    def forward(self, input, target, n_clusters):
        #_assert_no_grad(target)
        return self._discriminative_loss(input, target, n_clusters)
    
    def _discriminative_loss(self, image, instances_bs, annid):
        for i in range(image.size(0)):
            img = image[i,:,:,:]
            instances = instances_bs[i]
            imshape = img.shape
            instshape = instances.shape
            #feats,clusters,h,w
            img = img.unsqueeze(1).expand(imshape[0],instshape[0],imshape[1],imshape[2])
            #1,clusters,h,w
            instances = instances.unsqueeze(0)
            
            
            mns = cluster_means(img,instances)
            cvar = cluster_vars(img,instances,mns)
            uniqids = np.unique(annid[i])
            mean_of_class = []
            
            for val in uniqids:
                indices = np.where(annid[i]==val)
                if len(indices[0]) > 1:
                    dist_intra = distances(mns,indices[0],self.delta_dist_intra)
                    var_intra = cvar[indices[0]].mean()
                    reg_term = torch.mean(torch.norm(mns[:,indices[0]],1,0))
                    mean_of_class.append(torch.t(mns[:,indices[0]].mean(dim=1).view(1,-1)))
                else:
                    dist_intra = torch.zeros(1)[0]
                    var_intra = cvar[indices[0]][0]
                    reg_term = torch.norm(mns[:,indices[0]],1,0)[0]
                    mean_of_class.append(mns[:,indices[0]])
                class_loss[val-1,0] += var_intra
                class_loss[val-1,1] += dist_intra
                class_loss[val-1,2] += reg_term
                class_loss[val-1,3] += torch.ones(1)[0]
            mns_mean = torch.stack(mean_of_class,dim=1)[:,:,0]
            class_dist += distances(mns_mean,np.arange(mns_mean.shape[1]),self.delta_dist_inter)

        scaled_loss = class_loss[:,:3]/(class_loss[:,3].unsqueeze(1).expand(40,3)+eps)
        loss = torch.mm(scaled_loss,nn.view(-1,1)).sum() + class_dist/len(image)
        return loss
    
    def cluster_means(img,instances):
        #feats,clusters,h,w
        result = img.float()*instances.float()
        #feats,clusters
        means = result.sum(dim=[2,3])/instances.sum(dim=[2,3]).float()
        return(means)

    def cluster_vars(self,img,instances,means):
        #feats,clusters,h*w
        mn_shape = means.shape
        means = means.unsqueeze(2).expand(mn_shape[0],mn_shape[1],256*256)
        means = means.view(mn_shape[0],mn_shape[1],256,256)
        var = (torch.clamp(torch.norm((img - means),1,0) - self.delta_var,min=0)**2) * instances[0,:,:,:].float()
        
        new_var = var.sum([1,2])/instances[0,:].float().sum([1,2])
        
        return new_var

    def distances(means,indices,delta_dist):
        
        if means.shape[1] > 1:
            
            ax1,ax2 = means.shape
            temp_ma = means[:,indices].unsqueeze(2).expand(ax1,len(indices),len(indices))
            temp_mb = temp_ma.permute(0,2,1)
            diff = temp_ma - temp_mb
            margin = 2*delta_dist*(1-torch.eye(len(indices)))
            c_dist = torch.sum(torch.clamp(margin-torch.norm(diff,1,0),min=0)**2)
            
            dist = c_dist/(len(indices)*(len(indices)-1))
        
        else:
            dist = 0
        return dist
