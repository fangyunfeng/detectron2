# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The original 3DCE, fusing features of neighboring slices
and only keep the feature map of the central slice"""
from torch import nn
from detectron2.config import configurable
from .build import META_ARCH_REGISTRY

__all__ = ["FeatureFusion3dce"]


@META_ARCH_REGISTRY.register()
class FeatureFusion3dce(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.num_image = cfg.INPUT.NUM_IMAGES_3DCE
        self.out_dim = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.in_dim = cfg.MODEL.BACKBONE.IN_CHANNELS

        self.conv = nn.Conv2d(self.num_image * self.in_dim, self.out_dim, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, fs, images=None):

        for pi in fs:
            _, c, w, h = fs[pi].shape
            x = fs[pi].reshape(-1, self.num_image * c, w, h)
            x = self.conv(x)
            fs[pi] = x

        if images is not None:
            images.tensor = images.tensor[int(self.num_image/2)::self.num_image]
            return fs, images
        else:
            print('Error: images is None')
