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
import os  
import time  
from datetime import datetime  

import pytorch_lightning as pl  
import torch  
import torch.distributed as dist  
from omegaconf import OmegaConf  
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger  

from datamodules.argovese_v1_datamodule import ArgoverseV1DataModule  
from models.drg import DRG  

def main(args):  
    """Arguments for running the baseline."""   
    cfg = OmegaConf.load(args.config_path)  

    pl.seed_everything(args.seed)  

    datamodule = ArgoverseV1DataModule(cfg.data_cfg, args)  

    model = DRG(cfg)  

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")  
    log_dir = f"log/{date_str}"  
    os.makedirs(log_dir, exist_ok=True)  
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="records")  
    csv_logger = CSVLogger(save_dir=log_dir, name='records')  

  
    checkpoint_callback = ModelCheckpoint(  
        monitor=args.monitor,  
        mode="min",  
        save_top_k=args.save_top_k,  
        dirpath=os.path.join(log_dir, "checkpoints"),  
        filename="{epoch}-{val_minade_k:.2f}",  
        every_n_epochs=2  
    )  
   
    trainer = pl.Trainer(  
        accelerator="gpu",  
        devices=args.num_gpus,  
        max_epochs=args.epochs,  
        logger=[tb_logger, csv_logger],  
        callbacks=[checkpoint_callback], 
        gradient_clip_val=cfg.opt_cfg.gradient_clip_val,  
        precision=16,   
        strategy="ddp_find_unused_parameters_true" if args.num_gpus > 1 else None,  
        num_nodes=args.num_nodes,  
    )  
 
    start_time = time.time()  
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint_path if args.checkpoint_path else None)  
    end_time = time.time()  
 
    training_time = (end_time - start_time) / 60  
    print(f"Training completed in {training_time:.2f} minutes")  


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--mode", default="train", type=str, choices=["train", "val", "test"], help="Mode: train/val/test")  
    parser.add_argument("--features_dir", default="", type=str, help="Data path")  
    parser.add_argument("--config_path", required=False, default='configs/Av1_cfg.yaml', type=str, help="Path to the config file (YAML)")  
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")  
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")  
    parser.add_argument("--data_aug", type=bool, default=False, help="Data augmentation")  
    parser.add_argument("--num_workers", type=int, default=32, help="Number of data loader workers")  
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  
    parser.add_argument('--monitor', type=str, default='val_minade_k', choices=['val_minade_k', 'val_minfde_k', 'val_mr_k'])  
    parser.add_argument('--save_top_k', type=int, default=3)  
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training")  
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")  
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint to resume training")  

    args = parser.parse_args()  
    main(args)