#!/usr/bin/env python

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.frozen_clip.vision_transformer import QuickGELU, Attention
from models.frozen_clip.weight_loaders import weight_loader_fn_dict
from models.frozen_clip.vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)


class EVLDecoder(nn.Module):

    def __init__(
        self,
        # spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        # enable_temporal_conv: bool = True,
        # enable_temporal_pos_embed: bool = True,
        # enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        # self.enable_temporal_conv = enable_temporal_conv
        # self.enable_temporal_pos_embed = enable_temporal_pos_embed
        # self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(num_layers)]
        )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        B, L, C = in_features[0]['out'].size()
        # N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)
        # x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']
            x = self.decoder_layers[i](x, frame_features)
            
        return x


class EVLTransformer(nn.Module):

    def __init__(
        self,
        backbone_name: str = 'ViT-B/16-lnpre',
        backbone_type: str = 'clip',
        backbone_path: str = 'pretrained/clip/ViT-B-16.pt',
        backbone_mode: str = 'freeze_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)
        self.backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_layers=decoder_num_layers,
            in_feature_dim=self.backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            mlp_dropout=decoder_mlp_dropout,
        )
    
    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[backbone_name])
        backbone.load_state_dict(state_dict, strict=True) # weight_loader_fn is expected to strip unused parameters
        
        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone] # avoid backbone parameter registration

        return vit_presets[backbone_name]


    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone


    def forward(self, x: torch.Tensor):
        backbone = self._get_backbone(x)
        
        # B, C, H, W = x.shape
        
        features = backbone(x)[-self.decoder_num_layers:] # get features of last n layers 
        # print("features", features[0]['out'])
        x = self.decoder(features)
        # x = self.proj(x[:, 0, :])
        
        return x[:, 0, :]