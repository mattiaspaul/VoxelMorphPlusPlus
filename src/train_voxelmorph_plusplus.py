# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os

H = 192
W = 192
D = 208



import numpy as np
import nibabel as nib
import struct
import scipy.ndimage
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
import sys
import time


import matplotlib.pyplot as plt
from utils_voxelmorph_plusplus import *


H = 192; W = 192; D = 208

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU()#nn.LeakyReLU(0.2)
        self.main2 = Conv(out_channels, out_channels, 1, stride,0)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.activation2 = nn.ReLU()#nn.LeakyReLU(0.2)


    def forward(self, x):
        out = self.activation(self.norm(self.main(x)))
        out = self.activation2(self.norm2(self.main2(out)))
        return out

inshape = (224//2,224//2,224//2)
unet_half_res=True
nb_unet_features=None; nb_unet_levels=None
unet_feat_mult=1; nb_unet_conv_per_level=1; int_steps=7; int_downsize=2
bidir=False; use_probs=False; src_feats=1; trg_feats=1; unet_half_res=False; unet_half_res=True

def default_unet_features():
    nb_features = [[32, 48, 48, 64], [64, 48, 48, 48, 48, 32, 64]]  #  encoder,decoder
    return nb_features


def evaluate(unet_model,heatmap,ii):
    gpu_usage()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            mind_fix = mind_all_fix[ii].cuda().half()
            mind_mov = mind_all_mov[ii].cuda().half()
            
            keypts_fix = keypts_all_fix[int(ii)].cuda()
            keypts_mov = keypts_all_mov[int(ii)].cuda()

            disp_gt = keypts_mov-keypts_fix
            
            input = F.pad(torch.cat((mind_fix,mind_mov),1),(4,4,8,8,8,8)).cuda()
            output = unet_model(input)[:,:,4:-4,4:-4,2:-2]

            idx = torch.arange(keypts_all_fix[int(ii)].shape[0])
            sample_xyz = keypts_all_fix[int(ii)][idx]
            sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
            #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
            disp_pred = heatmap(sampled.permute(2,1,0,3,4))
#            disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

            pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)

    #plt.plot(pred_xyz[:,2].data.cpu(),disp_gt[:,2].data.cpu(),'.')
    tre0 = disp_gt.mul(100).pow(2).sum(1).sqrt().mean().item()
    tre1 = (disp_gt-pred_xyz).mul(100).pow(2).sum(1).sqrt().mean().item()
    return tre0,tre1,pred_xyz





if __name__ == "__main__":
    gpu_id = 0
    fold_nu = 1
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        if(i==1):
            print(f"Fold {i:>6}: {arg}")
            fold_nu  = int(arg)
        else:
            if(i==2):
                print(f"GPU ID {i:>6}: {arg}")
                gpu_id = int(arg)
            else:
                print(f"Argument {i:>6}: {arg}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(torch.cuda.get_device_name())
    
    #discretised heatmap grid (you may have to adapt the capture range from .3)
    mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11)).half()

    patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3))

    
    H = W = 192; D = 208
    nomind = False
    keypts_all_mov,keypts_all_fix,mind_all_mov,mind_all_fix,img_all_mov,img_all_fix,mask_all_mov,mask_all_fix = get_datasets(nomind)

    mind_all_mov2 = mind_all_mov.clone().detach()
    mind_all_fix2 = mind_all_fix.clone().detach()
    #mind_all_mov2 are 12-channel MIND-SSC features used for instance optimisation

    nomind = True
    keypts_all_mov,keypts_all_fix,mind_all_mov,mind_all_fix,img_all_mov,img_all_fix,mask_all_mov,mask_all_fix = get_datasets(nomind)
    #mind_all_mov are simply CT intensities

    unet_model = Unet(ConvBlock,inshape,infeats=2,nb_features=nb_unet_features,nb_levels=nb_unet_levels,\
            feat_mult=unet_feat_mult,nb_conv_per_level=nb_unet_conv_per_level,half_res=unet_half_res,)
    unet_model.cuda()



    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,7),nn.InstanceNorm3d(16),nn.ReLU(),\
                            nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),\
                            nn.ReLU(),nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),\
                            nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                
                            nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                            nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))
    heatmap.cuda()

    grid_sp = 2



    #keypoint heatmap loss
    if(fold_nu==1):
        idx_test = torch.tensor([5,7,15,17,18,20])
    if(fold_nu==2):
        idx_test = torch.tensor([4,6,14,16,19,21]) #fold2
    if(fold_nu==3):
        idx_test = torch.tensor([3,11,13,22,24,26])
    if(fold_nu==4):
        idx_test = torch.tensor([0,2,8,10,23,25])#fold4
    if(fold_nu==5):
        idx_test = torch.tensor([1,9,12,27,28,29])


    a_cat_b, counts = torch.cat([idx_test, torch.arange(30)]).unique(return_counts=True)
    idx_train = a_cat_b[torch.where(counts.eq(1))]
    print('train ids',idx_train)


    optimizer = torch.optim.Adam(list(unet_model.parameters())+list(heatmap.parameters()),lr=0.001)#0.001
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,700,0.5)
    t0 = time.time()
    run_tre = torch.empty(0,1); run_tre_test = torch.empty(0,1); 
    run_loss = torch.zeros(4900)
    for i in range(4900):
        ii = idx_train[torch.randperm(len(idx_train))[:1]]
        keypts_fix = keypts_all_fix[int(ii)].cuda()
        keypts_mov = keypts_all_mov[int(ii)].cuda()
        mind_fix = mind_all_fix[ii].cuda().half()
        mind_mov = mind_all_mov[ii].cuda().half()
        fixed_mask = mask_all_fix[ii].view(1,1,H,W,D).cuda().half()
        moving_mask = mask_all_mov[ii].view(1,1,H,W,D).cuda().half()

        #Affine augmentation of images *and* keypoints 
        if(i%2==0):
            A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
            affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
            keypts_fix = torch.solve(torch.cat((keypts_fix,torch.ones(keypts_fix.shape[0],1).cuda()),1).t(),\
                                     torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0))[0].t()[:,:3]
            fixed_mask = F.grid_sample(mask_all_fix[ii].view(1,1,H,W,D).cuda().half(),affine.half())
            fixed_img = F.grid_sample(img_all_fix[ii].view(1,1,H,W,D).cuda().half(),affine.half())
            with torch.cuda.amp.autocast():
                mind_fix_ = (fixed_mask.float()*fixed_img.float()/500).half()
                mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)

        if(i%2==1):
            A = (torch.randn(3,4)*.035+torch.eye(3,4)).cuda()
            affine = F.affine_grid(A.unsqueeze(0),(1,1,H,W,D))
            keypts_mov = torch.solve(torch.cat((keypts_mov,torch.ones(keypts_mov.shape[0],1).cuda()),1).t(),\
                                     torch.cat((A,torch.tensor([0,0,0,1]).cuda().view(1,-1)),0))[0].t()[:,:3]
            moving_mask = F.grid_sample(mask_all_mov[ii].view(1,1,H,W,D).cuda().half(),affine.half())
            moving_img = F.grid_sample(img_all_mov[ii].view(1,1,H,W,D).cuda().half(),affine.half())
            with torch.cuda.amp.autocast():
                mind_mov_ = (moving_mask.float()*moving_img.float()/500).half()
                mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)

        disp_gt = keypts_mov-keypts_fix

        scheduler.step()
        optimizer.zero_grad()
        idx = torch.randperm(keypts_all_fix[int(ii)].shape[0])[:512]

        with torch.cuda.amp.autocast():
            #VoxelMorph requires some padding
            input = F.pad(torch.cat((mind_fix,mind_mov),1),(4,4,8,8,8,8)).cuda()
            output = unet_model(input)[:,:,4:-4,4:-4,2:-2]

            sample_xyz = keypts_fix[idx]#keypts_all_fix[int(ii)][idx]#fix
            #todo nearest vs bilinear
            #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
            sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
            #disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(512,-1,3,3,3))
            disp_pred = heatmap(sampled.permute(2,1,0,3,4))


            pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
            loss = nn.MSELoss()(pred_xyz,disp_gt[idx])
            
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss[i] = loss.item()
        if(i%150==19):
            print(i,run_loss[i-18:i-1].mean(),time.time()-t0,'sec')
        if(i%200==99):
            #note: online evaluation is without instance optimisation (hence worse keypoint error)
            ii = idx_test[0:1]
            tre0,tre1,_ = evaluate(unet_model,heatmap,ii)
            run_tre = torch.cat((run_tre,torch.tensor([tre1]).view(-1,1)),0)
            print('training: tre0','%0.3f'%tre0,'tre1','%0.3f'%tre1)
            tre0,tre1,_ = evaluate(unet_model,heatmap,ii)
            run_tre_test = torch.cat((run_tre_test,torch.tensor([tre1]).view(-1,1)),0)
            print('test(EMPIRE): tre0','%0.3f'%tre0,'tre1','%0.3f'%tre1)
            ii = idx_test[2:3]
            tre0,tre1,_ = evaluate(unet_model,heatmap,ii)
            run_tre_test = torch.cat((run_tre_test,torch.tensor([tre1]).view(-1,1)),0)
            print('test(COPD): tre0','%0.3f'%tre0,'tre1','%0.3f'%tre1)
            ii = idx_test[5:6]
            tre0,tre1,_ = evaluate(unet_model,heatmap,ii)
            run_tre_test = torch.cat((run_tre_test,torch.tensor([tre1]).view(-1,1)),0)
            print('test(L2R): tre0','%0.3f'%tre0,'tre1','%0.3f'%tre1)
    
    #plt.plot(F.avg_pool1d(F.avg_pool1d(run_loss[:].view(1,1,-1),11,stride=1),11,stride=1).squeeze())
    unet_model.cpu(); heatmap.cpu()
    models = {'unet_model':unet_model.state_dict(),'heatmap':heatmap.state_dict()}
    torch.save(models,'l2r_lung_ct/repeat_l2r_voxelmorph_heatmap_keypoint_fold'+str(fold_nu)+'.pth')
    unet_model.cuda(); heatmap.cuda()
    #plt.show()
    #plt.plot(run_tre[:i],'s-')
    #plt.plot(run_tre_test[:i].reshape(-1,3).mean(1),'o-')


    print('done')






