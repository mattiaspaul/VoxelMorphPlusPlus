#!/usr/bin/env python

from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(torch.cuda.get_device_name())
import math
import struct
import csv
import time


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



#discretised heatmap grid (you may have to adapt the capture range from .3)
mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda()*.3,(1,1,11,11,11),align_corners=False).half()

patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0).cuda().half(),(1,1,3,3,3),align_corners=False)


H = W = 192; D = 208

def get_data(fixed_file,moving_file,fixed_mask_file,moving_mask_file):
    H = 192
    W = 192
    D = 208
    
    mind_ch = 12
    ii = 0

    time_mind = 0

    
    mask_fix = torch.from_numpy(nib.load(fixed_mask_file).get_fdata()).float().view(1,1,H,W,D)
    mask_mov = torch.from_numpy(nib.load(moving_mask_file).get_fdata()).float().view(1,1,H,W,D)
    zfill = 2
    img_fix = torch.from_numpy(nib.load(fixed_file).get_fdata()).float()
    img_mov = torch.from_numpy(nib.load(moving_file).get_fdata()).float()
    img_fix += 1000 #important that scans are in HU 
    img_mov += 1000

    grid_sp = 2

    #compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        with torch.cuda.amp.autocast():

            mind_fix_ = mask_fix.view(1,1,H,W,D).cuda().half()*\
            MINDSSC(img_fix.view(1,1,H,W,D).cuda(),1,2).half()
            mind_mov_ = mask_mov.view(1,1,H,W,D).cuda().half()*\
            MINDSSC(img_mov.view(1,1,H,W,D).cuda(),1,2).half()
            mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)
            mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)

        
        
    return mind_fix,mind_mov,img_fix,img_mov,mask_fix,mask_mov
    
def main(fixed_file,moving_file,fixed_mask_file,moving_mask_file,net_model_file,warped_file,disp_file,case_number):
    grid_sp = 2


    mind_fix,mind_mov,img_fix,img_mov,mask_fix,mask_mov = get_data(fixed_file,moving_file,fixed_mask_file,moving_mask_file)

    img_fix2 = F.avg_pool3d((mask_fix*img_fix).view(1,1,H,W,D).float()/500,grid_sp,stride=grid_sp)
        
    img_mov2 = F.avg_pool3d((mask_mov*img_mov).view(1,1,H,W,D).float()/500,grid_sp,stride=grid_sp)
        
        
    in_channel = 2
    unet_model = Unet(ConvBlock,inshape,infeats=in_channel,nb_features=nb_unet_features,\
                      nb_levels=nb_unet_levels,feat_mult=unet_feat_mult,nb_conv_per_level=nb_unet_conv_per_level,\
                      half_res=unet_half_res,)
    heatmap = nn.Sequential(nn.ConvTranspose3d(64,16,7),nn.InstanceNorm3d(16),nn.ReLU(),\
                            nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                            nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear',align_corners=False),\
                            nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),\
                            nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.Conv3d(32,16,3,padding=1),\
                            nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))

#'l2r_lung_ct/repeat_l2r_voxelmorph_heatmap_keypoint_fold'+str(fold_nu)+'.pth'
    models = torch.load(net_model_file)
    heatmap.load_state_dict(models['heatmap'])
    unet_model.load_state_dict(models['unet_model'])
    unet_model.cuda(); heatmap.cuda()

    print('starting Voxelmorph++ registration of '+fixed_file+' and '+moving_file)
#mask_fix = mask_all_fix[ii].cuda()
            #print('validation random')

    torch.cuda.synchronize()
    t0 = time.time()
    

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            img_fix2 = img_fix2.cuda().half()
            img_mov2 = img_mov2.cuda().half()
            
            #keypts_fix = keypts_fix.cuda()
            #keypts_mov = keypts_mov.cuda()
            keypts_rand = 2*torch.rand(2048*24,3).cuda()-1
            val = F.grid_sample(mask_fix.cuda(),keypts_rand.view(1,-1,1,1,3),align_corners=False)
            idx1 = torch.nonzero(val.squeeze()==1).reshape(-1)
            
            keypts_fix = keypts_rand[idx1[:1024*2]]
            
            
            input = F.pad(torch.cat((img_fix2,img_mov2),1),(4,4,8,8,8,8)).cuda()
            output = unet_model(input)[:,:,4:-4,4:-4,2:-2]

            sample_xyz = keypts_fix
            sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear',align_corners=False)
            #sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
            disp_pred = heatmap(sampled.permute(2,1,0,3,4))
#            disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

            pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


    dense_flow_ = thin_plate_dense(keypts_fix.unsqueeze(0), pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
    dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2

    torch.cuda.synchronize()
    t1 = time.time()
    t_inf = t1-t0; t0=t1;


    disp_hr = dense_flow

    if(True):
        grid_sp = 2

        mind_fix = mind_fix.cuda().half()
        mind_mov = mind_mov.cuda().half()

        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)
        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = .65# with tps: .5, without:0.7
        for iter in range(50):#80
            optimizer.zero_grad()
            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),\
                                                    3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()
            scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),\
                                              align_corners=False,mode='bilinear')#,padding_mode='border')
            sampled_cost = (patch_mov_sampled-mind_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()
        fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
        disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)

    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1),\
                                            3,padding=1,stride=1),3,padding=1,stride=1)


    disp_hr = torch.flip(disp_smooth/torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1).cuda()*2,[1])

    pred_xyz = F.grid_sample(disp_hr.float(),keypts_fix.cuda().view(1,-1,1,1,3),mode='bilinear',\
                             align_corners=False).squeeze().t()
    torch.cuda.synchronize()
    t1 = time.time()
    t_adam = t1-t0; t0=t1;

    print('run time','%0.3f'%t_inf,'sec (net)','%0.3f'%t_adam,'sec (adam)')

    if(disp_file is not None):
        torch.save(fitted_grid.data.cpu(),disp_file)
    
    # compute_tre if possible
    if(case_number is not None):
        i = int(case_number)-1
        copd_lms = torch.load('copd_converted_lms.pth')

        disp = F.grid_sample(disp_hr.cpu(),copd_lms['lm_copd_exh'][i].view(1,-1,1,1,3),align_corners=False).squeeze().t()
        tre_before = ((copd_lms['lm_copd_exh'][i]-copd_lms['lm_copd_insp'][i])*(torch.tensor([208/2,192/2,192/2]).view(1,3)*torch.tensor([1.25,1.,1.75]))).pow(2).sum(1).sqrt()

        tre_after=((copd_lms['lm_copd_exh'][i]-copd_lms['lm_copd_insp'][i]+disp)*(torch.tensor([208/2,192/2,192/2]).view(1,3)*torch.tensor([1.25,1.,1.75]))).pow(2).sum(1).sqrt()
        print('DIRlab COPD #',case_number,'TRE before (mm)','%0.3f'%(tre_before.mean().item()),\
              'TRE after (mm)','%0.3f'%(tre_after.mean().item()))
    warped = F.grid_sample(img_mov.view(1,1,H,W,D),disp_hr.cpu().permute(0,2,3,4,1)+\
                           F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D),align_corners=False),align_corners=False).cpu()
    if(warped_file is not None):
        nib.save(nib.Nifti1Image((warped-1000).data.squeeze().numpy(),np.diag([1.75,1.25,1.75,1])),warped_file)
    
    if((warped_file is None)&(disp_file is None)):
        img_fix *= mask_fix.view_as(img_fix)
        warped *= mask_fix.view_as(warped)
        
        plt.imshow(torch.clamp(img_fix.squeeze().cpu()[:,110],0,700).div(700).pow(1.25).t().flip(0),'Blues')
        plt.imshow(torch.clamp(warped.squeeze().squeeze()[:,110],0,500).div(500).pow(1.25).t().flip(0),'Oranges',alpha=.5)
        plt.axis('off')
        plt.savefig('voxelmorph_plusplus_warped.png')
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-F', '--fixed_file', required=True, help="fixed scan (exhale) nii.gz")
    parser.add_argument('-M', '--moving_file', required=True, help="moving scan (inspiration) nii.gz")
    parser.add_argument('-f', '--fixed_mask_file', required=True, help="mask fixed nii.gz")
    parser.add_argument('-m', '--moving_mask_file', required=True, help="mask moving nii.gz")

    parser.add_argument('-n', '--net_model_file', required=True, help="network models pth-file")

    parser.add_argument('-w', '--warped_file', required=False, help="output nii.gz file")
    parser.add_argument('-d', '--disp_file', required=False, help="output displacements pth-file")
    parser.add_argument('-c', '--case_number', required=False, help="DIRlab COPD case number (for TRE)")



    main(**vars(parser.parse_args()))

