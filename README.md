# VoxelMorph++
## Going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation

The recent Learn2Reg challenge shows that single-scale U-Net architectures (e.g. VoxelMorph) with spatial transformer loss,  do not generalise well beyond the cranial vault and fall short of state-of-the-art performance for abdominal or intra-patient lung registration.

**VoxelMorph++** takes two straightforward steps that greatly reduce this gap in accuracy.
1. we employ keypoint self-supervision with a novel heatmap prediction network head
2. we replace multiple learned fine-tuning steps by a single instance optimisation with hand-crafted features and the Adam optimiser. 

**VoxelMorph++** robustly estimates large deformations using the discretised heatmaps and unlike PDD-Net does not require a fully discretised architecture with correlation layer. If no keypoints are available heatmaps can still be used in an unsupervised using only a nonlocal MIND metric. 

We outperform VoxelMorph by improving nonlinear alignment by 77% compared to 22% - reaching target registration errors of 2 mm on the DIRlab-COPD dataset. Extending the method to semantic features sets new stat-of-the-art performance of 70% on inter-subject abdominal CT registration. 

![](wbir2022_voxelmorph2.png | width=300)
