from torch import nn, Tensor
from math import gcd
from torch.nn import functional as F
import torch
from torch.nn import MultiheadAttention
from typing import Any, Dict, List, Tuple, Union, Optional

import torch  
import torch.nn as nn  
from torch import Tensor  
from typing import Optional, Tuple  

import torch  
import torch.nn as nn  
from torch import Tensor  
from models.layers import RP_GAFormer


"Lane-Actor Fusion Network"
class LAFusionNet(nn.Module):
    def __init__(self, config):
        super(LAFusionNet, self).__init__()
        self.cfg = config
        d_embed = self.cfg.d_embed
        dropout = self.cfg.dropout
        update_edge = self.cfg.update_edge

        self.proj_actor = nn.Sequential(
            nn.Linear(self.cfg.d_actor, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(self.cfg.d_lane, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(self.cfg.d_rpe_in, self.cfg.d_rpe),
            nn.LayerNorm(self.cfg.d_rpe),
            nn.ReLU(inplace=True)
        )
        self.fuse_scene = RP_GAFormer(num_layers=3, 
                                    in_dim=128, 
                                    key_dim=64, 
                                    value_dim=64, 
                                    num_heads=4, 
                                    dropout=0.1, 
                                    ffn_dim=256) 
        
    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Tensor):
        
        

        actors = self.proj_actor(actors) 
        lanes = self.proj_lane(lanes) 

        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpe in zip(actor_idcs, lane_idcs, rpe_prep):

            _actors = actors[a_idcs] 
            _lanes = lanes[l_idcs] 
            scene_features = torch.cat([_actors, _lanes], dim=0)  
            
            out = self.fuse_scene(scene_features, rpe) 
            
            actors_new.append(out[:len(a_idcs)]) 
            lanes_new.append(out[len(a_idcs):]) 
        
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        
        return actors, lanes