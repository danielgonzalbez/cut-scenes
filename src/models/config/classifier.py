from dataclasses import dataclass

@dataclass
class CNNConfig:
    last_dim:int = 64
    channels:list[int] = (128,96,64,48,32)
    dims:list[int] = (400,384,256,128)
    dropout: float = 0.2


@dataclass
class ModelConfig:
    inner_dim:int = 48
    depth_attn:int = 3
    num_heads: int = 2
    head_dim: int = 48
    time_dropout: float = 0.2
    audio_dropout: float = 0.4
    final_dropout: float = 0.4
    time_dim: int = 16
    seq_len: int=None




