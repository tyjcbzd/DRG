# Copyright (c) 2025, Yi Tang. All rights reserved.  
#  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
#     http://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.  
import argparse  

import pytorch_lightning as pl  
from pytorch_lightning.callbacks import ModelCheckpoint  
from omegaconf import OmegaConf  

from datamodules.argovese_v1_datamodule import ArgoverseV1DataModule  
from models.drg import DRG  
import torch  

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger   
import time  
from datetime import datetime  
import os  

def main():  
    """Arguments for running the baseline."""  
    parser = argparse.ArgumentParser()  
  
    parser.add_argument("--ckpt_path", default ='', type=str, help="Path to the checkpoint file for testing")  
    parser.add_argument("--mode", default="val", type=str)
    parser.add_argument("--features_dir", default="", type=str, help="Data path")
    parser.add_argument("--config_path", required=False, default='configs/Av1_cfg.yaml', type=str, help="Path to the config file (YAML)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers") 
    parser.add_argument("--use_cuda", default=True, help="Use CUDA if available")  
    
    args = parser.parse_args()  

    cfg = OmegaConf.load(args.config_path)  

    pl.seed_everything(args.seed) 
    
    if args.use_cuda and torch.cuda.is_available():  
        device = torch.device("cuda", 0)  
    else:  
        device = torch.device('cpu') 
    
    datamodule = ArgoverseV1DataModule(cfg.data_cfg, args, args.mode)  

    model = DRG 
    csv_logger = CSVLogger("logs/", name="my_test")  
    
    if args.ckpt_path:  
        model = model.load_from_checkpoint(args.ckpt_path, cfg=cfg, device=device) 
    else:  
        raise ValueError("You must specify a checkpoint path for testing using --ckpt_path")  

    trainer = pl.Trainer(  
        devices=[0],  
        precision = 16, 
        logger=csv_logger 
    )  

    start_time = time.time()  

    trainer.validate(model, datamodule=datamodule)  
    
    end_time = time.time()  
    testing_time = (end_time - start_time) / 60  
    print(f"Testing completed in {testing_time:.2f} minutes")  


if __name__ == "__main__":  
    main()