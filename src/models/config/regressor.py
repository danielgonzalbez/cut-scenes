from dataclasses import dataclass


@dataclass
class CNNConfig:
    last_dim:int = 32
    channels:list[int] = (128,256,128,64,32)
    dims:list[int] = (100,128,64,48)
    dropout:float = 0
    

@dataclass
class ModelConfig:
  lstm_layers: int = 2
  input_size: int = 32
  audio_dim: int = 32
  bidirectional: bool = False
  time_dim: int = 16
  audio_dropout: float = 0
  time_dropout: float=0
  seq_len: int = 4
