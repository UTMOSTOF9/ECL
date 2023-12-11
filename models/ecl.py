#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: ecl.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 15:54
'''

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

from .cutout import Cutout


class balanced_proxies(nn.Module):
    def __init__(self, dim, proxy_num=3):
        super(balanced_proxies, self).__init__()
        protos = torch.nn.Parameter(torch.empty([proxy_num, dim]))
        torch.nn.init.xavier_uniform_(protos, gain=1)
        self.proxies = protos

    def forward(self):
        centers = F.normalize(self.proxies, dim=-1)
        return centers


class ECL_model(nn.Module):
    def __init__(self, num_classes=8, feat_dim=512):
        super(ECL_model, self).__init__()
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(cnns, ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'])
        self.num_classes = num_classes

        self.clip_feature_extractor = clip.load('ViT-B/32')[0]

        dimension = 512*4

        self.clip_dimension_adjust = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # self.feat_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.feat_fusion = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feat_dim * 2, dim_feedforward=feat_dim * 4, nhead=8, batch_first=True),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        self.fc = nn.Linear(dimension, self.num_classes)
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        self.generate_head = nn.Conv2d(32, 3, 3, 1, 1)
        self.cutout = Cutout(1, 4, fill_value=0)

    def forward(self, x):
        if isinstance(x, list):
            feat1 = self.backbone(x[0])['avgpool']
            b = feat1.shape[0]
            feat1 = feat1.view(b, -1)
            feat1_mlp = self.head(feat1)
            logits = self.fc(feat1)

            feat2 = self.backbone(x[1])['avgpool']
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_mlp = self.head(feat2)

            # with clip
            clip_feat1 = self.clip_dimension_adjust(self.clip_feature_extractor.encode_image(x[0]))
            clip_feat2 = self.clip_dimension_adjust(self.clip_feature_extractor.encode_image(x[1]))
            fusion_feat1 = F.normalize(self.feat_fusion(torch.cat([feat1_mlp, clip_feat1], dim=-1)))
            fusion_feat2 = F.normalize(self.feat_fusion(torch.cat([feat2_mlp, clip_feat2], dim=-1)))

            feats = self.backbone(self.cutout(x[0]))
            reconstruct_maps1 = self.generate_head(self.fpn(feats)['layer1'])

            feats = self.backbone(self.cutout(x[1]))
            reconstruct_maps2 = self.generate_head(self.fpn(feats)['layer1'])

            return logits, [fusion_feat1, fusion_feat2], [reconstruct_maps1, reconstruct_maps2]

        else:
            feat1 = self.backbone(x)['avgpool']
            feat1 = feat1.view(feat1.shape[0], -1)

            logits = self.fc(feat1)

            return logits


class ECL_swin_model(nn.Module):
    def __init__(self, num_classes=8, feat_dim=512):
        super().__init__()

        self.backbone = create_model('swinv2_base_window12_192', pretrained=True, features_only=True)
        self.num_classes = num_classes

        dimension = 1024
        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dimension, self.num_classes)
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], 16)
        self.generate_head = nn.Conv2d(32, 3, 3, 1, 1)
        self.cutout = Cutout(1, 4, fill_value=0)

    def forward(self, x):
        if isinstance(x, list):
            b = x[0].shape[0]
            feat1 = self.avgpool(self.backbone(x[0])[-1].permute(0, 3, 1, 2))
            feat1 = feat1.view(b, -1)
            feat1_mlp = F.normalize(self.head(feat1))
            logits = self.fc(feat1)

            feat2 = self.avgpool(self.backbone(x[1])[-1].permute(0, 3, 1, 2))
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_mlp = F.normalize(self.head(feat2))

            feats = self.backbone(self.cutout(x[0]))
            feats = {f'layer{i}': x.permute(0, 3, 1, 2) for i, x in enumerate(feats)}
            reconstruct_maps1 = self.generate_head(self.fpn(feats)['layer1'])

            feats = self.backbone(self.cutout(x[1]))
            feats = {f'layer{i}': x.permute(0, 3, 1, 2) for i, x in enumerate(feats)}
            reconstruct_maps2 = self.generate_head(self.fpn(feats)['layer1'])
            return logits, [feat1_mlp, feat2_mlp], [reconstruct_maps1, reconstruct_maps2]

        else:
            feat1 = self.backbone(x)
            feat1 = feat1.view(feat1.shape[0], -1)

            logits = self.fc(feat1)

            return logits


def build_model(name, **kwargs):
    if name == 'ResNet50':
        return ECL_model(**kwargs)
    elif name == 'SwinV2':
        return ECL_swin_model(**kwargs)
    else:
        raise NotImplementedError
