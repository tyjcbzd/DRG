import pytorch_lightning as pl
import torch
from models.encoder import Encoder
from models.lafn import LAFusionNet
from models.decoder import MLPDecoder

from losses.reg_cls_loss import LossFunc
import numpy as np
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
import torch.nn as nn
from pprint import pprint 
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
import warnings

class DRG(pl.LightningModule):

    def __init__(self, cfg):
        super(DRG, self).__init__()
        self.save_hyperparameters() 
        self.net_cfg = cfg.net_cfg
        self.loss_cfg = cfg.loss_cfg
        self.eval_cfg = cfg.eval_cfg
        self.opt_cfg = cfg.opt_cfg  
        # Model structure
        self.encoder = Encoder( in_actor=self.net_cfg.in_actor, 
                                d_actor=self.net_cfg.d_actor,
                                in_lane=self.net_cfg.in_lane,
                                d_lane=self.net_cfg.d_lane,
                                n_fpn_scale=self.net_cfg.n_fpn_scale, 
                                dropout=self.net_cfg.dropout)
        self.lafn = LAFusionNet(config=self.net_cfg)
        self.decoder = MLPDecoder(config=self.net_cfg)
        
        # Loss
        self.cal_loss = LossFunc(self.loss_cfg)
        
        # for validate
        self._validate_result = []
        # for test
        self._test_results = []
        
    def forward(self, batch):
        actors, actor_idcs, lanes, lane_idcs, rpe_ga = batch['ACTORS'], batch['ACTOR_IDCS'], batch['LANES'], batch['LANE_IDCS'], batch['RPE_GA'] 
        rpe_actors, rpe_lanes, rpe_actor_lane =  rpe_ga['actors'], rpe_ga['lanes'], rpe_ga['actors_lanes']
        # print("0")
        # # rebuild rpe for scene
        # rpe = []
        # for i in range(len(rpe_scene)):
        #     rpe.append({'scene':rpe_scene[i],
        #                 'scene_mask':rpe_scene_mask[i],})
        actors, lanes = self.encoder(actors, lanes, actor_idcs, lane_idcs, rpe_actors, rpe_lanes)
        # print(actors.shape)
        # print(lanes.shape)
        actors, lanes = self.lafn(actors, actor_idcs, lanes, lane_idcs, rpe_actor_lane)
        out = self.decoder(actors, actor_idcs)

        return out
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        gt_preds, pad_flags=batch["TRAJS_FUT"], batch["PAD_FUT"]
        loss_out = self.cal_loss(out, gt_preds, pad_flags)

        self.log("train_epoch_reg_loss", loss_out['reg_loss'], sync_dist=True)
        self.log("train_epoch_cls_loss", loss_out['cls_loss'], sync_dist=True)
        self.log("train_epoch_loss", loss_out['loss'], sync_dist=True)

        return loss_out

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        gt_preds, pad_flags=batch["TRAJS_FUT"], batch["PAD_FUT"]
        loss_out = self.cal_loss(out, gt_preds, pad_flags)

        self.log("train_epoch_loss", loss_out['loss'], sync_dist=True)
        
        # post process
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0) 
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out['out_raw'] = out
        post_out['traj_pred'] = reg  
        post_out['prob_pred'] = cls  
        
        traj_pred = post_out['traj_pred'] 
        prob_pred = post_out['prob_pred'] 

        traj_fut = torch.stack([traj[0, :, 0:2] for traj in batch['TRAJS_FUT']]) 

        # to np.ndarray
        traj_pred = np.asarray(traj_pred.cpu().detach().numpy()[:, :, :, :2], np.float32)
        prob_pred = np.asarray(prob_pred.cpu().detach().numpy(), np.float32)
        traj_fut = np.asarray(traj_fut.cpu().detach().numpy(), np.float32)

        seq_id_batch = batch['SEQ_ID']
        batch_size = len(seq_id_batch)

        pred_dict = {}
        gt_dict = {}
        prob_dict = {}
        
        for j in range(batch_size): 
            seq_id = seq_id_batch[j]
            pred_dict[seq_id] = traj_pred[j]
            gt_dict[seq_id] = traj_fut[j]
            prob_dict[seq_id] = prob_pred[j]

        res_1 = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 1, self.eval_cfg.g_pred_len, miss_threshold=self.eval_cfg.miss_thres, forecasted_probabilities=prob_dict)
        # # Max #guesses (K): 6
        res_k = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 6, self.eval_cfg.g_pred_len, miss_threshold=self.eval_cfg.miss_thres, forecasted_probabilities=prob_dict)

        eval_out = {}
        eval_out['minade_1'] = res_1['minADE']
        eval_out['minfde_1'] = res_1['minFDE']
        eval_out['mr_1'] = res_1['MR']
        eval_out['brier_fde_1'] = res_1['brier-minFDE']

        eval_out['minade_k'] = res_k['minADE']
        eval_out['minfde_k'] = res_k['minFDE']
        eval_out['mr_k'] = res_k['MR']
        eval_out['brier_fde_k'] = res_k['brier-minFDE']

        for j in range(batch_size):
            # Store the sample name and its metrics  
            result = {  
                "seq_id": seq_id,  
                "minade_1": eval_out["minade_1"][j],  
                "minfde_1": eval_out["minfde_1"][j],  
                "mr_1": eval_out["mr_1"][j],  
                "minade_k": eval_out["minade_k"][j],  
                "minfde_k": eval_out["minfde_k"][j],  
                "mr_k": eval_out["mr_k"][j],  
            }  
            self._validate_result.append(result)  
        
        self.log_dict(  
            {f"val_{key}": val.item() if isinstance(val, torch.Tensor) else val for key, val in eval_out.items()},    # 这里加上了前缀，可以删除掉
            on_step=False,   
            on_epoch=True,   
            prog_bar=True,
            sync_dist=True  
        )  

        return eval_out
        

    def configure_optimizers(self):
        lr_scale = 1.0 
        if self.opt_cfg.lr_scale_func == "linear":  
            lr_scale = self.trainer.world_size  
        elif self.opt_cfg.lr_scale_func == "sqrt":  
            lr_scale = np.sqrt(self.trainer.world_size)  
        
        lr = self.opt_cfg.init_lr * lr_scale  

        optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=lr, 
                                         weight_decay=self.opt_cfg.weight_decay)
        
        values = [x * lr_scale for x in self.opt_cfg.lr_values]  
        scheduler = PolylineLR(optimizer, milestones=self.opt_cfg.milestones, values=values) 

        if scheduler:  
            return {  
                "optimizer": optimizer,  
                "lr_scheduler": {  
                    "scheduler": scheduler,  
                    "interval": "epoch",  # or 'step'  
                    "frequency": 1  
                }  
            }  
        return optimizer
    
class PolylineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, values, last_epoch=-1, verbose=False):

        assert len(milestones) == len(values), '[PolylineLR] length must be same'
        assert all(x >= 0 for x in milestones), '[PolylineLR] milestones must be positive'
        assert all(x >= 0 for x in values), '[PolylineLR] values must be positive'
        assert milestones[0] == 0, '[PolylineLR] milestones must start from 0'
        assert milestones == sorted(milestones), '[PolylineLR] milestones must be in ascending order'

        self.milestones = milestones
        self.values = values
        self.n_intervals = len(self.milestones) - 1
        super(PolylineLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        lr = self._get_value(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]

    def _get_value(self, epoch):
        assert epoch >= 0
        for i in range(self.n_intervals):
            e_lb = self.milestones[i]
            e_ub = self.milestones[i + 1]
            if epoch < e_lb or epoch >= e_ub:
                continue  # not in this interval
            v_lb = self.values[i]
            v_ub = self.values[i + 1]
            v_e = (epoch - e_lb) / (e_ub - e_lb) * (v_ub - v_lb) + v_lb
            return v_e
        return self.values[-1]