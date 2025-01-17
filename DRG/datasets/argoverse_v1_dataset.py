import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
#
import torch
from torch.utils.data import Dataset


class ArgoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        df = pd.read_pickle(self.dataset_files[idx])
        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID'] 
        city_name = data['CITY_NAME'] 
        orig = data['ORIG'] 
        rot = data['ROT'] 

        trajs = data['TRAJS'] 

        trajs_obs = trajs[:, :self.obs_len]

        trajs_fut = trajs[:, self.obs_len:]

        pad_flags = data['PAD_FLAGS'] 
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        trajs_ctrs = data['TRAJS_CTRS'] 
        trajs_vecs = data['TRAJS_VECS'] 


        graph = data['LANE_GRAPH']
        

        lane_ctrs = graph['lane_ctrs'] 
        lane_vecs = graph['lane_vecs'] 

     
        rpes = dict()
        scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0) 
        scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0) 
        rpes['actors_lanes'] = self.compute_actor_lane_rpe(scene_ctrs, scene_vecs) 
        rpes['actors'] = self.compute_actor_actor_rpe(torch.from_numpy(trajs_ctrs), torch.from_numpy(trajs_vecs))
        rpes['lanes'] = self.compute_lane_lane_rpe(torch.from_numpy(lane_ctrs),torch.from_numpy(lane_vecs))
        

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['TRAJS_OBS'] = trajs_obs
        data['TRAJS_FUT'] = trajs_fut
        data['PAD_OBS'] = pad_obs
        data['PAD_FUT'] = pad_fut
        data['TRAJS_CTRS'] = trajs_ctrs
        data['TRAJS_VECS'] = trajs_vecs
        data['LANE_GRAPH'] = graph
        data['RPE'] = rpes

        return data

    def _get_cos(self, v1, v2):
        
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_dang

    def _get_sin(self, v1, v2):
        
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_dang
    
    def compute_actor_actor_rpe(self, actors_ctrs, actors_vecs):
        d_pos_actor = (actors_ctrs.unsqueeze(0) - actors_ctrs.unsqueeze(1)).norm(dim=-1)
        d_pos_actor = d_pos_actor * 2 / 100.0  
        cos_a1_actor = self._get_cos(actors_vecs.unsqueeze(0), actors_vecs.unsqueeze(1))
        sin_a1_actor = self._get_sin(actors_vecs.unsqueeze(0), actors_vecs.unsqueeze(1))
        
        v_pos_actor = actors_ctrs.unsqueeze(0) - actors_ctrs.unsqueeze(1)
        cos_a2_actor = self._get_cos(actors_vecs.unsqueeze(0), v_pos_actor)
        sin_a2_actor = self._get_sin(actors_vecs.unsqueeze(0), v_pos_actor)

        ang_rpe_actor = torch.stack([cos_a1_actor, sin_a1_actor, cos_a2_actor, sin_a2_actor])
        rpe_actor = torch.cat([ang_rpe_actor, d_pos_actor.unsqueeze(0)], dim=0)
        return rpe_actor


    def compute_lane_lane_rpe(self, lane_ctrs, lane_vecs):
        d_pos_lane = (lane_ctrs.unsqueeze(0) - lane_ctrs.unsqueeze(1)).norm(dim=-1)
        d_pos_lane = d_pos_lane * 2 / 100.0  

        cos_a1_lane = self._get_cos(lane_vecs.unsqueeze(0), lane_vecs.unsqueeze(1))
        sin_a1_lane = self._get_sin(lane_vecs.unsqueeze(0), lane_vecs.unsqueeze(1))
        
        v_pos_lane = lane_ctrs.unsqueeze(0) - lane_ctrs.unsqueeze(1)
        cos_a2_lane = self._get_cos(lane_vecs.unsqueeze(0), v_pos_lane)
        sin_a2_lane = self._get_sin(lane_vecs.unsqueeze(0), v_pos_lane)

        ang_rpe_lane = torch.stack([cos_a1_lane, sin_a1_lane, cos_a2_lane, sin_a2_lane])
        rpe_lane = torch.cat([ang_rpe_lane, d_pos_lane.unsqueeze(0)], dim=0)
        return rpe_lane


    def compute_actor_lane_rpe(self, ctrs, vecs, radius=100.0): 

        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        d_pos = d_pos * 2 / radius 
        pos_rpe = d_pos.unsqueeze(0)
      
        cos_a1 = self._get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
        sin_a1 = self._get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
        
        v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1) 
        cos_a2 = self._get_cos(vecs.unsqueeze(0), v_pos)
        sin_a2 = self._get_sin(vecs.unsqueeze(0), v_pos)

        ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
        rpe_actor_lane = torch.cat([ang_rpe, pos_rpe], dim=0)
        return rpe_actor_lane

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        batch = from_numpy(batch)
        data = dict()
        data['BATCH_SIZE'] = len(batch)
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
     
       
        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS_OBS'], data['PAD_OBS'])
        lanes, lane_idcs = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])

        rpes_ga = self.rpe_gather(data['RPE'])
        
        data['ACTORS'] = actors
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANE_IDCS'] = lane_idcs
        data['RPE_GA'] = rpes_ga
        return data

    def actor_gather(self, batch_size, actors, pad_flags):
        num_actors = [len(x) for x in actors] 

        act_feats = []
        for i in range(batch_size):
           
            vel = torch.zeros_like(actors[i])
            vel[:, 1:, :] = actors[i][:, 1:, :] - actors[i][:, :-1, :] 
            act_feats.append(torch.cat([vel, pad_flags[i].unsqueeze(2)], dim=2)) 
        act_feats = [x.transpose(1, 2) for x in act_feats] 
        actors = torch.cat(act_feats, 0)  
        actor_idcs = []  
        count = 0
        for i in range(batch_size):
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        return actors, actor_idcs

    def graph_gather(self, batch_size, graphs):
        
        lane_idcs = list()
        lane_count = 0
        for i in range(batch_size):
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idcs.append(l_idcs)
            lane_count = lane_count + graphs[i]["num_lanes"]

        graph = dict()
        for key in ["node_ctrs", "node_vecs", "turn", "control", "intersect", "left", "right"]:
            graph[key] = torch.cat([x[key] for x in graphs], 0)
        for key in ["lane_ctrs", "lane_vecs"]:
            graph[key] = [x[key] for x in graphs]

        lanes = torch.cat([graph['node_ctrs'],
                           graph['node_vecs'],
                           graph['turn'],
                           graph['control'].unsqueeze(2),
                           graph['intersect'].unsqueeze(2),
                           graph['left'].unsqueeze(2),
                           graph['right'].unsqueeze(2)], dim=-1) 

        return lanes, lane_idcs

    def rpe_gather(self, rpes_list):
        
        batch_size = len(rpes_list)  
        
        scene_rpes = []  
        scene_masks = []  
        actor_rpes = []  
        lane_rpes = []  
        actor_lane_rpes = []  
        

        for rpe in rpes_list:  
            
            actor_rpes.append(rpe['actors'])  
            lane_rpes.append(rpe['lanes'])  
            actor_lane_rpes.append(rpe['actors_lanes'])  

        batched_rpes = {  
            'actors': actor_rpes,  
            'lanes': lane_rpes,  
            'actors_lanes': actor_lane_rpes  
        }  
        
        return batched_rpes 
        
    def data_augmentation(self, df):

        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data
        
        data['TRAJS_CTRS'][..., 1] *= -1
        data['TRAJS_VECS'][..., 1] *= -1
        data['TRAJS'][..., 1] *= -1

        data['LANE_GRAPH']['lane_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['lane_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['node_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['node_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']

        return data
