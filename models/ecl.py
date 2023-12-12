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

from .cross_attention import CrossAttention
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

        clip_feature_extractor, _ = clip.load('ViT-B/32')
        for param in clip_feature_extractor.parameters():
            param.requires_grad = False

        self.clip_feature_extractor = clip_feature_extractor.eval()

        dimension = 512*4

        self.ssl_branch = nn.ModuleDict(
            {
                'fpn': FeaturePyramidNetwork([256, 512, 1024, 2048], 16),
                'generate_head': nn.Conv2d(16, 3, 3, 1, 1),
            }
        )
        self.clip_branch = nn.ModuleDict(
            {
                'clip_dimension_adjust': nn.Linear(512, dimension),
                'feat_fusion': nn.TransformerEncoderLayer(
                    d_model=dimension*2,
                    nhead=8,
                    dim_feedforward=dimension*4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                )
            }
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.fusion_weight = nn.Parameter(torch.ones(2), requires_grad=True)
        self.classifier = nn.Linear(dimension, self.num_classes)
        self.flatten = nn.Flatten()
        self.cutout = Cutout(1, 4, fill_value=0)

    def forward(self, x):
        if isinstance(x, list):
            feat1 = self.flatten(self.backbone(x[0])['avgpool'])
            feat2 = self.flatten(self.backbone(x[1])['avgpool'])

            # classifier
            logits = self.classifier(feat1)

            # clip
            with torch.no_grad():
                clip_feat1 = self.clip_feature_extractor.encode_image(x[0])
                clip_feat2 = self.clip_feature_extractor.encode_image(x[1])

            clip_feat1 = self.clip_branch['clip_dimension_adjust'](clip_feat1)
            clip_feat2 = self.clip_branch['clip_dimension_adjust'](clip_feat2)

            w = self.fusion_weight
            fusion_feat1 = F.normalize(self.fusion_head(w[0] * clip_feat1 + w[1] * feat1))
            fusion_feat2 = F.normalize(self.fusion_head(w[0] * clip_feat2 + w[1] * feat2))

            # fusion_feat1 = F.normalize(self.clip_branch['feat_fusion'](torch.cat((clip_feat1, feat1), dim=-1)))
            # fusion_feat2 = F.normalize(self.clip_branch['feat_fusion'](torch.cat((clip_feat2, feat2), dim=-1)))

            # ssl
            feats1 = self.backbone(self.cutout(x[0]))
            feats2 = self.backbone(self.cutout(x[1]))
            reconstruct_maps1 = self.ssl_branch['generate_head'](self.ssl_branch['fpn'](feats1)['layer1'])
            reconstruct_maps2 = self.ssl_branch['generate_head'](self.ssl_branch['fpn'](feats2)['layer1'])

            return logits, [fusion_feat1, fusion_feat2], [reconstruct_maps1, reconstruct_maps2]

        else:
            feat1 = self.flatten(self.backbone(x)['avgpool'])
            logits = self.classifier(feat1)
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
