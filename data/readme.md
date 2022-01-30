##Data preprocessing is described in the paper
please also see /src/utils_voxelmorph_plusplus.py for details

The exhale lung CT scans are resampled to 1.75 x 1.25 x 1.75 mm, the inspiration ones to 1.75 x 1.00 x 1.25 mm. Next the volumes are cropped to a region-of-interest of size 192 x 192 x 208 that centres the lung masks. This already may half the TRE for DIR-Lab COPD as it peforms a translation + scale global transformation.

The keypoint correspondences are obtained by running http://www.mpheinrich.de/code/corrFieldWeb.zip please cite if you use this tool
[2] Heinrich, M. P., Handels, H., & Simpson, I. J. A. (2015). Estimating large lung motion in COPD patients by symmetric regularised correspondence fields. MICCAI 2015 Springer LNCS, 9350, 338–345. https://doi.org/10.1007/978-3-319-24571-3_41



