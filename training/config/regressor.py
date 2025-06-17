import torch
import torch.nn as nn

num_epochs: int = 40
lr: float = 8e-4
weight_decay: float = 3e-3
weighted_sampling: bool = False # if we have an unbalanced dataset, we can do weighted sampling
weighted_loss: bool = True # if we have an unbalanced dataset, we can do weighted loss
alpha_weight_loss: float = 0.65
train_batch_size: int = 128
test_batch_size: int = 128
dtype = torch.bfloat16
n_mels: int = 128
log_wandb: bool=False
grad_accum_steps: int = 1
epochs_warmup: int = 5
load_checkpoint: bool = False
checkpoint_file: str = None
run_name: str = None
project_name: str = None