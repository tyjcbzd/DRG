from torch import nn, Tensor
from math import gcd
from torch.nn import functional as F
import torch
from torch.nn import MultiheadAttention
from typing import Any, Dict, List, Tuple, Union, Optional
import math
import numpy as np


class MLPDecoder(nn.Module):
    def __init__(self,
                 config) -> None:
        super(MLPDecoder, self).__init__()
        self.config = config
        self.hidden_size = self.config.d_embed 
        self.future_steps = self.config.g_pred_len 
        self.num_modes = self.config.g_num_modes 
        self.param_out = self.config.param_out  
        self.N_ORDER = self.config.param_order

        dim_mm = self.hidden_size * self.num_modes 
        dim_inter = dim_mm // 2 
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True)
        )

        self.mat_T = self._get_T_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps)
        self.mat_Tp = self._get_Tp_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps)

        self.reg = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts ** i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(self,
                embed: torch.Tensor,
                actor_idcs: List[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mat_T = self.mat_T.to(embed.device)
        mat_Tp = self.mat_Tp.to(embed.device)
        # input embed: [159, 128]
        embed = self.multihead_proj(embed).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2) 
        # print('embed: ', embed.shape)  

        cls = self.cls(embed).view(self.num_modes, -1).permute(1, 0)  
        cls = F.softmax(cls * 1.0, dim=1)  

        
        param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2) 
        param = param.permute(1, 0, 2, 3)  
        reg = torch.matmul(mat_T, param)  
        vel = torch.matmul(mat_Tp, param[:, :, 1:, :]) / (self.future_steps * 0.1)
        
        res_cls, res_reg, res_aux = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i] 
            res_cls.append(cls[idcs]) 
            res_reg.append(reg[idcs]) 

        return res_cls, res_reg, res_aux