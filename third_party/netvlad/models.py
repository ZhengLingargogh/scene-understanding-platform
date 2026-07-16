"""VGG + NetVLAD (vendored from ibl, no relative imports)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init


class VGG(nn.Module):
    def __init__(self, depth: int = 16, pretrained: bool = False, train_layers: str = "conv5") -> None:
        super().__init__()
        fix_layers = {"conv5": 24, "conv4": 17, "conv3": 10, "conv2": 5, "full": 0}
        vgg = torchvision.models.vgg16(weights=None)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.feature_dim = 512
        if pretrained:
            for layer in layers[: fix_layers[train_layers]]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.base(x)
        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)
        return pool_x, x


def vgg16(**kwargs) -> VGG:
    return VGG(16, **kwargs)


class NetVLAD(nn.Module):
    def __init__(self, num_clusters: int = 64, dim: int = 512, normalize_input: bool = True) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x).view(x.shape[0], self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(x.shape[0], self.dim, -1)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - (
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        )
        residual *= soft_assign.unsqueeze(2)
        return residual.sum(dim=-1)


class EmbedNetPCA(nn.Module):
    def __init__(self, base_model: VGG, net_vlad: NetVLAD, dim: int = 4096) -> None:
        super().__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.pca_layer = nn.Conv2d(net_vlad.num_clusters * net_vlad.dim, dim, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.base_model(x)
        vlad_x = self.net_vlad(x)
        vlad_x = F.normalize(vlad_x, p=2, dim=2)
        vlad_x = vlad_x.view(x.size(0), -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=1)
        vlad_x = vlad_x.view(vlad_x.size(0), -1, 1, 1)
        vlad_x = self.pca_layer(vlad_x).view(vlad_x.size(0), -1)
        return F.normalize(vlad_x, p=2, dim=-1)
