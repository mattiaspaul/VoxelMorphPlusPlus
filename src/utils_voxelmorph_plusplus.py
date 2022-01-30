import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math
import struct
import csv
import time

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

    
def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist
    
def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    # kernel size
    kernel_size = radius * 2 + 1
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    
    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean()*0.001, mind_var.mean()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind

def mind_loss(x, y):
    return torch.mean( (MINDSSC(x) - MINDSSC(y)) ** 2 )


def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def pdist2(x, y, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = (y**2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device
    
    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    return ind, dist*A, A

def laplacian(kpts, k, lambd, sigma=0):
    _, dist, A = knn_graph(kpts, k)
    W = lambd * A.squeeze(0)
    if sigma > 0:
        W = W * torch.exp(- dist.squeeze(0) / (sigma ** 2))
    return (torch.diag(W.sum(1) + 1) - W).unsqueeze(0), W.unsqueeze(0)




def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

def default_unet_features():
    nb_features = [[32, 48, 48, 64],  # encoder
        [64, 48, 48, 48, 48, 32, 64]] #decoder
    return nb_features


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,ConvBlock,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
    
def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n+4, n+4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.solve(v, A)[0]
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()
    
def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2**12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D//step, H//step, W//step
    
    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)
    
    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1*H1*W1
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)
        
    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
    
    return y2


def get_datasets(nomind=False):
    H = 192
    W = 192
    D = 208

    import csv
    import struct
    import time

    keypts_all_mov = []
    keypts_all_fix = []

    mind_ch = 12
    if(nomind):
        mind_ch = 1

    mind_all_mov = torch.zeros(30,mind_ch,H//2,W//2,D//2).pin_memory()
    mind_all_fix = torch.zeros(30,mind_ch,H//2,W//2,D//2).pin_memory()

    img_all_mov = torch.zeros(30,1,H,W,D).pin_memory()
    img_all_fix = torch.zeros(30,1,H,W,D).pin_memory()
    mask_all_mov = torch.zeros(30,1,H,W,D).pin_memory()
    mask_all_fix = torch.zeros(30,1,H,W,D).pin_memory()

    time_mind = 0

    for ii,i in enumerate((1,7,8,14,18,20,21,28,1,2,3,4,5,6,7,8,9,10,1,2,3,5,6,8,9,11,12,14,17,19)):
        if(ii<8):
            folder = 'EMPIRE10/'; dat = '.dat';
        else:
            folder = 'COPDgene/'; dat = '_insp.dat';
        if(ii>=18):
            folder = 'l2r_lung_ct/training/scans/';
        if(ii<18):
            mask_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_exp_mask.nii.gz').get_fdata()).float()
            mask_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(2)+'_insp_mask.nii.gz').get_fdata()).float()
        if(ii>=18):
            mask_all_fix[ii] = torch.from_numpy(nib.load('l2r_lung_ct/training/lungMasks/case_'+str(i).zfill(3)+'_exp.nii.gz').get_fdata()).float()
            mask_all_mov[ii] = torch.from_numpy(nib.load('l2r_lung_ct/training/lungMasks/case_'+str(i).zfill(3)+'_insp.nii.gz').get_fdata()).float()
        zfill = 2
        if(ii>=18):
            zfill = 3
        img_all_fix[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_exp.nii.gz').get_fdata()).float()
        img_all_mov[ii] = torch.from_numpy(nib.load(folder+'/case_'+str(i).zfill(zfill)+'_insp.nii.gz').get_fdata()).float()
        if(ii<18):
            img_all_fix[ii] += 1000
            img_all_mov[ii] += 1000

        if(ii<18):
            with open(folder+'/keypoints/case_'+str(i).zfill(2)+dat, 'rb') as content_file:
                content = content_file.read()
            corrfield = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(-1,6).float()
        else:
            corrfield = torch.empty(0,6)
            with open('l2r_lung_ct/keypoints/case_0'+str(i).zfill(2)+'.csv', newline='') as csvfile:
                fread = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in fread:
                    corrfield = torch.cat((corrfield,torch.from_numpy(np.array(row).astype('float32')).float().view(1,6)),0)

        keypts_fix = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1).cuda()
        keypts_mov = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1).cuda()
        if((ii>=8)&(ii<18)):
            keypts_mov = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1).cuda()
            keypts_fix = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1).cuda()

        keypts_all_mov.append(keypts_mov)
        keypts_all_fix.append(keypts_fix)

        mean_mask = F.grid_sample(mask_all_fix[ii:ii+1],keypts_fix.view(1,-1,1,1,3).cpu()).mean()+F.grid_sample(mask_all_mov[ii:ii+1],keypts_mov.view(1,-1,1,1,3).cpu()).mean()
        if(mean_mask<1.97):
            print(ii,i,'mean_mask',mean_mask)
        grid_sp = 2

        torch.cuda.synchronize()
        t0 = time.time()
        #compute MIND descriptors and downsample (using average pooling)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if(nomind):
                    mind_fix_ = mask_all_fix[ii].view(1,1,H,W,D).cuda().float()*\
                    (img_all_fix[ii].view(1,1,H,W,D).cuda().float()/500)
                    mind_mov_ = mask_all_mov[ii].view(1,1,H,W,D).cuda().float()*\
                    (img_all_mov[ii].view(1,1,H,W,D).cuda().float()/500)
                
                else:
                    mind_fix_ = mask_all_fix[ii].view(1,1,H,W,D).cuda().half()*\
                    MINDSSC(img_all_fix[ii].view(1,1,H,W,D).cuda(),1,2).half()
                    mind_mov_ = mask_all_mov[ii].view(1,1,H,W,D).cuda().half()*\
                    MINDSSC(img_all_mov[ii].view(1,1,H,W,D).cuda(),1,2).half()
                mind_fix = F.avg_pool3d(mind_fix_,grid_sp,stride=grid_sp)
                mind_mov = F.avg_pool3d(mind_mov_,grid_sp,stride=grid_sp)
        torch.cuda.synchronize()
        t1 = time.time()
        time_mind += (t1-t0)
        mind_all_fix[ii] = mind_fix.cpu()
        mind_all_mov[ii] = mind_mov.cpu()
    print('mind computation',time_mind/30,'sec') 
    return keypts_all_mov,keypts_all_fix,mind_all_mov,mind_all_fix,img_all_mov,img_all_fix,mask_all_mov,mask_all_fix 