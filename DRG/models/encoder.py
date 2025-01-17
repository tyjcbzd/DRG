import torch 
from models.layers import Res1d, Conv1d, PointAggregateBlock
from torch import nn, Tensor
from torch.nn import functional as F
from models.layers import RP_GAFormer

class Encoder(nn.Module):
    def __init__(self, in_actor, 
                       d_actor,
                       in_lane,
                       d_lane,
                       n_fpn_scale, 
                       dropout):
        super(Encoder,self).__init__()

        self.aa_encoder = AAEncoder(n_in=in_actor,
                                    hidden_size=d_actor,
                                    n_fpn_scale=n_fpn_scale)
        self.ll_encoder = LLEncoder(in_size=in_lane, 
                                    hidden_size=d_lane, 
                                    dropout=dropout)

    def forward(self, actors_feats, lanes_feats, actor_idcs, lane_idcs, rpe_actors, rpe_lanes):
        actors = self.aa_encoder(actors_feats, actor_idcs, rpe_actors)
        lanes = self.ll_encoder(lanes_feats, lane_idcs, rpe_lanes)
        
        return actors, lanes 


class AAEncoder(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in, hidden_size, n_fpn_scale):
        super(AAEncoder, self).__init__()
        
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale 
        num_blocks = [2] * n_fpn_scale # [2,2,2,2]

        groups = []
        for i in range(len(num_blocks)): 
            group = []
            if i == 0: 
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)): 
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

 
        self.gru = nn.GRU(  
            input_size=hidden_size,  
            hidden_size=hidden_size,  
            num_layers=4,  
            batch_first=True,   
        )  
        
       
        self.proj_rpe_actor = nn.Sequential(
             nn.Linear(5, hidden_size),
             nn.LayerNorm(hidden_size),
             nn.ReLU(inplace=True)  
        )


        self.graph_interaction = RP_GAFormer(num_layers=1, 
                                           in_dim=128, 
                                           key_dim=64, 
                                           value_dim=64, 
                                           num_heads=1, 
                                           dropout=0.1, 
                                           ffn_dim=256) 

        
        self.fusion = nn.Sequential(  
            nn.Linear(hidden_size * 2, hidden_size),  
            nn.ReLU(inplace=True),  
            nn.LayerNorm(hidden_size)  
        )  

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, actors: Tensor, actor_idcs, rpe_actors: Tensor) -> Tensor:
        
        
        out = actors
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)  

        out = self.lateral[-1](outputs[-1]) 
        for i in range(len(outputs) - 2, -1, -1): 
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False) # 
            out += self.lateral[i](outputs[i])
        
        actor_features, _ = self.gru(out.permute(0,2,1)) 
        actor_features = actor_features[:,-1,:]
        
        final_outputs = []
        for idcs, rpe_actor in zip(actor_idcs,rpe_actors): 
            _actors_feats = actor_features[idcs] 
            # print(_actors_feats.shape)
            # print(rpe_actor.shape)
            actors_rpe_attn = self.graph_interaction(_actors_feats, rpe_actor)  
            fused_features = self.norm(_actors_feats + actors_rpe_attn)  

            final_outputs.append(fused_features) 
        actors_rpes = torch.cat(final_outputs, dim=0)
        return actors_rpes
    
class LLEncoder(nn.Module):
    def __init__(self, in_size=10, hidden_size=128, dropout=0.1):
        super(LLEncoder, self).__init__()
       
        self.lane_proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre = PointAggregateBlock(hidden_size=hidden_size, aggre_out=True, dropout=dropout)

        self.cross_atten = RP_GAFormer(num_layers=1, 
                                    in_dim=128, 
                                    key_dim=64, 
                                    value_dim=64, 
                                    num_heads=1, 
                                    dropout=0.1, 
                                    ffn_dim=256) 
        
        
        # Layer Norms  
        self.norm = nn.LayerNorm(hidden_size)  
         
        self.feature_fusion = nn.Sequential(  
            nn.Linear(hidden_size * 2, hidden_size * 2),  
            nn.LayerNorm(hidden_size * 2),  
            nn.ReLU(),  
            nn.Linear(hidden_size * 2, hidden_size)  
        )  

    def forward(self, feats, lane_idcs, rpe_lanes): 
       
        x = self.lane_proj(feats)  
        lane_features = self.aggre(x)  
        final_outputs= []
        for idcs, rpe_lane in zip(lane_idcs, rpe_lanes):
            _lane_feats = lane_features[idcs] # 174,128
            lane_rpe_attn = self.cross_atten(_lane_feats, rpe_lane)
            fused_features = self.norm(_lane_feats + lane_rpe_attn)  

            final_outputs.append(fused_features) 
            
        lanes_rpes = torch.cat(final_outputs, dim=0)
        return lanes_rpes