from dataclasses import dataclass

@dataclass
class CNNConfig:
    last_dim:int = 384
    channels:list[int] = (128,256,384,256,64)
    dims:list[int] = (400,512,768,512)
    dropout:float = 0


@dataclass
class ModelConfig:
    inner_dim:int = 48
    depth_attn:int = 3
    num_heads: int = 2
    head_dim: int = 48
    drop_prev_samples: float = 0.2
    drop_proj: float = 0.2




