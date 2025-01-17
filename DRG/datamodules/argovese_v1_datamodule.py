from importlib import import_module  
from torch.utils.data import DataLoader  
from pytorch_lightning import LightningDataModule  
from typing import Optional  
from datasets.argoverse_v1_dataset import ArgoDataset
import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler  


class ArgoverseV1DataModule(LightningDataModule):  
    def __init__(self, cfg, args, verbose=False):  
        super().__init__()  

        self.cfg = cfg

        self.args = args  
        self.verbose = verbose  

        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers


    def print(self, *info):  
        if self.verbose:  
             print("[ArgoverseV1DataModule]", *info)  

    def setup(self, stage: Optional[str] = None):  
        self.print('[ArgoverseV1DataModule] setup')  
        train_dir = self.args.features_dir + 'train/'  
        val_dir = self.args.features_dir + 'val1/'  
        test_dir = self.args.features_dir + 'test/'  

        if stage == "fit" or stage is None:  
            self.train_dataset = ArgoDataset(train_dir, 'train',  
                                             pred_len=self.cfg.g_pred_len,  
                                             verbose=self.verbose, 
                                             aug=self.args.data_aug)  
            self.val_dataset = ArgoDataset(val_dir, 'val',  
                                           obs_len=self.cfg.g_obs_len,  
                                           pred_len=self.cfg.g_pred_len,  
                                           verbose=self.verbose, 
                                           aug=False)  
        if stage == "validate" or stage is None:
            self.val_dataset = ArgoDataset(val_dir, 'val',  
                                           obs_len=self.cfg.g_obs_len,  
                                           pred_len=self.cfg.g_pred_len,  
                                           verbose=self.verbose, 
                                           aug=False)  
        if stage == "test" or stage is None:  
            self.test_dataset = ArgoDataset(test_dir, 'test',  
                                            obs_len=self.cfg.g_obs_len,  
                                            pred_len=self.cfg.g_pred_len,  
                                            verbose=self.verbose)   
    def train_dataloader(self):  
        if dist.is_available() and dist.is_initialized(): 
            sampler = DistributedSampler(self.train_dataset)  
            shuffle = False 
        else:  
            sampler = None  
            shuffle = True 

        return DataLoader(self.train_dataset, batch_size=self.batch_size,  
                          num_workers=self.num_workers, sampler=sampler,  
                          collate_fn=self.train_dataset.collate_fn, shuffle=shuffle)  

    def val_dataloader(self):  
        if dist.is_available() and dist.is_initialized():  
            sampler = DistributedSampler(self.val_dataset) if self.val_dataset else None  
        else:  
            sampler = None  
        return DataLoader(self.val_dataset, batch_size=self.batch_size,  
                          num_workers=self.num_workers, sampler=sampler,  
                          collate_fn=self.val_dataset.collate_fn, shuffle=False)  


    def test_dataloader(self):  
        if dist.is_available() and dist.is_initialized():  
            sampler = DistributedSampler(self.test_dataset) if self.test_dataset else None  
        else:  
            sampler = None  
        return DataLoader(self.test_dataset, batch_size=self.batch_size,  
                          num_workers=self.num_workers, sampler=sampler,  
                          collate_fn=self.test_dataset.collate_fn, shuffle=False) 
    