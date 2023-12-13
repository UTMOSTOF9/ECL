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
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

from .cutout import Cutout
from .frozen_clip.finetune_clip import EVLTransformer


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
        self.backbone = torch.nn.Sequential(*(list(cnns.children())[:-1]))

        self.num_classes = num_classes

        dimension = 512*4

        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.fc = nn.Linear(dimension,self.num_classes)

    def forward(self,x):
        if isinstance(x, list):
            feat1 = self.backbone(x[0])
            feat1 = feat1.view(feat1.shape[0],-1)
            feat1_mlp = F.normalize(self.head(feat1))
            logits = self.fc(feat1)

            feat2 = self.backbone(x[1])
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_mlp = F.normalize(self.head(feat2))

            return logits,[feat1_mlp,feat2_mlp], None

        else:
            feat1 = self.backbone(x)
            feat1 = feat1.view(feat1.shape[0], -1)

            logits = self.fc(feat1)

            return logits


class ECL_EVL_model(nn.Module):
    def __init__(
        self, 
        num_classes:int=8, 
        feat_dim:int=512, # 128
        cls_dropout:float=0.5, 
        cnn_f:float=0.2,
        clip_f:float=0.8,
    ):
        super(ECL_EVL_model, self).__init__()
        self.num_classes = num_classes
        self.cnn_f = cnn_f
        self.clip_f = clip_f
        
        # backbone
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(cnns, ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'])
        backbone_feature_dim = 512*4
        
        self.backbone_head = nn.Sequential(
            nn.Linear(backbone_feature_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.backbone_fc = nn.Linear(backbone_feature_dim, self.num_classes)
        
        # clip
        self.clip_feature_extractor = EVLTransformer()
        clip_feature_dim = self.clip_feature_extractor.backbone_feature_dim # danger ops
        
        self.clip_head = nn.Sequential(
            nn.LayerNorm(clip_feature_dim),
            nn.Linear(clip_feature_dim, backbone_feature_dim),
        )
        
        # self training
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        self.generate_head = nn.Conv2d(32, 3, 3, 1, 1)
        self.cutout = Cutout(1, 4, fill_value=0)

    def forward(self, x):
        if isinstance(x, list):
            # with clip
            clip_feat1 = self.clip_head(self.clip_feature_extractor(x[0]))
            clip_feat2 = self.clip_head(self.clip_feature_extractor(x[1]))
            
            feat1 = self.backbone(x[0])['avgpool']
            feat1 = feat1.view(feat1.shape[0], -1) # dimension
            # fusion
            fusion_feat1 = self.cnn_f * feat1 + self.clip_f * clip_feat1
            feat1_mlp = F.normalize(self.backbone_head(fusion_feat1))
            logits = self.backbone_fc(fusion_feat1)

            feat2 = self.backbone(x[1])['avgpool']
            feat2 = feat2.view(feat2.shape[0], -1)
            # fusion
            fusion_feat2 = self.cnn_f * feat2 + self.clip_f * clip_feat2
            feat2_mlp = F.normalize(self.backbone_head(fusion_feat2))

            # fusion_feat1 = F.normalize(self.feat_fusion(torch.cat([feat1_mlp, clip_feat1], dim=-1)))
            # fusion_feat2 = F.normalize(self.feat_fusion(torch.cat([feat2_mlp, clip_feat2], dim=-1)))

            feats = self.backbone(self.cutout(x[0]))
            reconstruct_maps1 = self.generate_head(self.fpn(feats)['layer1'])
            feats = self.backbone(self.cutout(x[1]))
            reconstruct_maps2 = self.generate_head(self.fpn(feats)['layer1'])

            return logits, [feat1_mlp, feat2_mlp], [reconstruct_maps1, reconstruct_maps2]

        else:
            feat1 = self.backbone(x)['avgpool']
            feat1 = feat1.view(feat1.shape[0], -1) # dimension
            # feat1_mlp = F.normalize(self.head(feat1))
            clip_feat1 = self.clip_head(self.clip_feature_extractor(x))
            fusion_feat1 = self.cnn_f * feat1 + self.clip_f * clip_feat1
            
            logits = self.backbone_fc(fusion_feat1)

            return logits


class ECL_clip_model(nn.Module):
    def __init__(self, num_classes=8, feat_dim=512):
        super(ECL_clip_model, self).__init__()
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = create_feature_extractor(cnns, ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'])
        self.num_classes = num_classes

        self.clip_feature_extractor = clip.load('ViT-B/32')[0]

        dimension = 512*4

        self.clip_dimension_adjust = nn.Sequential(
            nn.Linear(512, dimension),
            nn.BatchNorm1d(dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension),
        )

        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # self.feat_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.feat_fusion = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=dimension * 2, dim_feedforward=dimension * 4, nhead=8, batch_first=True),
            nn.BatchNorm1d(dimension * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dimension * 2, dimension),
        )

        self.fc = nn.Linear(dimension, self.num_classes)
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        self.generate_head = nn.Conv2d(32, 3, 3, 1, 1)
        self.cutout = Cutout(1, 4, fill_value=0)

    def forward(self, x):
        if isinstance(x, list):
            
            clip_feat1 = self.clip_dimension_adjust(self.clip_feature_extractor.encode_image(x[0]))
            clip_feat2 = self.clip_dimension_adjust(self.clip_feature_extractor.encode_image(x[1]))
            
            feat1 = self.backbone(x[0])['avgpool']
            feat1 = feat1.view(feat1.shape[0], -1)
            fusion_feat1 = self.feat_fusion(torch.cat([feat1, clip_feat1], dim=-1))
            feat1_mlp = F.normalize(self.head(fusion_feat1))
            logits = self.fc(fusion_feat1)

            feat2 = self.backbone(x[1])['avgpool']
            feat2 = feat2.view(feat2.shape[0], -1)
            fusion_feat2 = self.feat_fusion(torch.cat([feat2, clip_feat2], dim=-1))
            feat2_mlp = F.normalize(self.head(fusion_feat2))
            
            feats = self.backbone(self.cutout(x[0]))
            reconstruct_maps1 = self.generate_head(self.fpn(feats)['layer1'])

            feats = self.backbone(self.cutout(x[1]))
            reconstruct_maps2 = self.generate_head(self.fpn(feats)['layer1'])

            return logits, [feat1_mlp, feat2_mlp], [reconstruct_maps1, reconstruct_maps2]

        else:
            feat1 = self.backbone(x)['avgpool']
            feat1 = feat1.view(feat1.shape[0], -1)
            clip_feat1 = self.clip_dimension_adjust(self.clip_feature_extractor.encode_image(x))
            fusion_feat1 = self.feat_fusion(torch.cat([feat1, clip_feat1], dim=-1))
            
            logits = self.fc(fusion_feat1)

            return logits


def build_model(name, **kwargs):
    if name == 'ResNet50':
        return ECL_clip_model(**kwargs)
    elif name == 'EVL':
        return ECL_EVL_model(**kwargs)
    else:
        raise NotImplementedError
