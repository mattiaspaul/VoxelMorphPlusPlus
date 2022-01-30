# VoxelMorph++
## Going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation

The recent Learn2Reg challenge shows that single-scale U-Net architectures (e.g. VoxelMorph) with spatial transformer loss,  do not generalise well beyond the cranial vault and fall short of state-of-the-art performance for abdominal or intra-patient lung registration.

**VoxelMorph++** takes two straightforward steps that greatly reduce this gap in accuracy.
1. we employ keypoint self-supervision with a novel heatmap prediction network head
2. we replace multiple learned fine-tuning steps by a single instance optimisation with hand-crafted features and the Adam optimiser. 

**VoxelMorph++** robustly estimates large deformations using the discretised heatmaps and unlike PDD-Net does not require a fully discretised architecture with correlation layer. If no keypoints are available heatmaps can still be used in an unsupervised using only a nonlocal MIND metric. 

We outperform VoxelMorph by improving nonlinear alignment by 77% compared to 22% - reaching target registration errors of 2 mm on the DIRlab-COPD dataset. Extending the method to semantic features sets new stat-of-the-art performance of 70% on inter-subject abdominal CT registration. Our network can be trained within 17 minutes on a single RTX A4000 with a carbon footprint of less than 20 grams.

![Overview figure](wbir2022_voxelmorph2.png)

## Implementation
We slightly adapt the basic Voxelmorph U-Net as a backbone baseline, adding some more convolutions and feature channels
```
unet_half_res=True
def default_unet_features():
    nb_features = [
        [32, 48, 48, 64],             # encoder
        [64, 48, 48, 48, 48, 32, 64]  # decoder
    ]
    return nb_features
unet_model = Unet(inshape,infeats=14,nb_features=nb_unet_features,nb_levels=nb_unet_levels,\
            feat_mult=unet_feat_mult,nb_conv_per_level=nb_unet_conv_per_level,half_res=unet_half_res,)

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

