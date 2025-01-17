import torch.nn as nn
from math import gcd
from torch.nn import functional as F
import torch

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out
    
class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class TransformerLayer(nn.Module):  
    def __init__(self, in_dim=128, key_dim=64, value_dim=64, num_heads=4, dropout=0.1, ffn_dim=256):  
        super().__init__()  
        self.num_heads = num_heads  

        self.query_fc = nn.ModuleList([nn.Linear(in_dim, key_dim) for _ in range(num_heads)])  
        self.key_fc = nn.ModuleList([nn.Linear(in_dim, key_dim) for _ in range(num_heads)])  
        self.value_fc = nn.ModuleList([nn.Linear(in_dim, value_dim) for _ in range(num_heads)])  
        
        self.rpe_proj = nn.Linear(5, 1)  

        self.output_proj = nn.Linear(value_dim * num_heads, in_dim)  
  
        self.ffn = nn.Sequential(  
            nn.Linear(in_dim, ffn_dim),  
            nn.ReLU(),  
            nn.Linear(ffn_dim, in_dim)  
        )  

        self.layer_norm1 = nn.LayerNorm(in_dim)  
        self.layer_norm2 = nn.LayerNorm(in_dim)  
        self.dropout = nn.Dropout(dropout)  

        self._init_weights()  

    def _init_weights(self):  
        
        for fc in (self.query_fc + self.key_fc + self.value_fc):  
            nn.init.kaiming_normal_(fc.weight)  

    def forward(self, x, rpe):  
        
        resid = x  

        multi_head_output = self._multi_head_attention(x)  
        multi_head_output = self._apply_rpe(multi_head_output, rpe)  

        multi_head_output = self.output_proj(multi_head_output)  # [Na, value_dim * num_heads] -> [Na, in_dim]

        x = self.layer_norm1(resid + self.dropout(multi_head_output))  

        resid = x 
        ffn_out = self.ffn(x)  
        output = self.layer_norm2(resid + self.dropout(ffn_out))  

        return output  

    def _multi_head_attention(self, x):  
        attention_outputs = []  

        for i in range(self.num_heads):  
            
            query = F.relu(self.query_fc[i](x))  # [Na, key_dim]  
            key = F.relu(self.key_fc[i](x))      # [Na, key_dim]  
            value = F.relu(self.value_fc[i](x))  # [Na, value_dim]  

            
            attention_scores = query.mm(key.t())  # (Na, key_dim) @ (key_dim, Na) = [Na, Na]  
            attention_scores = attention_scores / (key.shape[-1] ** 0.5)  
            attention_weights = F.softmax(attention_scores, dim=1)  # [Na, Na]  

          
            attention_output = attention_weights.mm(value)  # [Na, Na] @ [Na, value_dim] = [Na, value_dim]  
            attention_outputs.append(attention_output)  


        multi_head_output = torch.cat(attention_outputs, dim=-1)  # [Na, value_dim]  
        return multi_head_output  

    def _apply_rpe(self, multi_head_output, rpe):  
       
        rpe_weights = self.rpe_proj(rpe.permute(1, 2, 0)).squeeze(-1)  # [5, Na, Na] -> [Na, Na]  
        adjusted_output = torch.matmul(rpe_weights, multi_head_output)  # [Na, Na] @ [Na, value_dim] = [Na, value_dim]  
        return adjusted_output  

''' Relative Position-aware Graph Attention Transformer (RP-GAT) '''
class RP_GAFormer(nn.Module):  
   
    def __init__(self, num_layers=4, in_dim=128, key_dim=64, value_dim=64, num_heads=4, dropout=0.1, ffn_dim=256):  
        super().__init__()  

        self.layers = nn.ModuleList([  
            TransformerLayer(in_dim, key_dim, value_dim, num_heads, dropout, ffn_dim)  
            for _ in range(num_layers)  
        ])  

    def forward(self, x, rpe):  
       
        for layer in self.layers:  
            x = layer(x, rpe)  
        return x