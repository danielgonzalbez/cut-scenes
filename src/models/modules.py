import torch.nn as nn
import torch
import math


class SwiGLU(nn.Module):
    def __init__(self, inner_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(inner_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)  
        x2 = self.linear2(x)  
        swish_x1 = x1 * self.sigmoid(x1)  
        return swish_x1 * x2 



class FeedForward(nn.Module):
  def __init__(self, in_dim:int, mid_dim:int, activation: nn.Module):
    super().__init__()
    self.norm = nn.RMSNorm(in_dim)
    self.linear1 = nn.Linear(in_dim, mid_dim)
    self.linear2 = nn.Linear(mid_dim, in_dim)
    self.act = activation

  def forward(self, x):
    x = self.norm(x)
    x_map = self.linear1(x)
    x_act = self.act(x_map)
    return self.linear2(x_act)
  


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim=16, num_heads=8, head_dim=4):
        super().__init__()

        self.hidden_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.scale_attn = self.hidden_dim ** (-0.5)  

       # define projections
        self.Q = nn.Linear(in_dim, self.hidden_dim)
        self.K = nn.Linear(in_dim, self.hidden_dim)
        self.V = nn.Linear(in_dim, self.hidden_dim)

        # final projection
        self.out = nn.Linear(self.hidden_dim, in_dim)

    def forward(self, key=None):
        # x shape: B x L x C
        B, L, C = key.shape
        q = self.Q(key).reshape(B, self.num_heads, L, self.head_dim)
        k = self.K(key).reshape(B, self.num_heads, L, self.head_dim)
        v = self.V(key).reshape(B, self.num_heads, L, self.head_dim)

        # self-attention:
        attn = q @ k.transpose(-1,-2) * self.scale_attn

        # with the expand, each value in amax will be repeated H*W times
        attn = attn - attn.amax(dim=-1)[:,:,:,None].expand(B, self.num_heads, L, L)

        weights = nn.functional.softmax(attn, dim=-1) # softmax for each row (B x num_heads x L x L)
        weighted_vals = (v.transpose(-1,-2) @ weights).reshape(B, L, self.hidden_dim)

        # output projection: B x L x hidden_dim -> B x L x in_dim
        return self.out(weighted_vals)
    


class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, in_dim:int, 
                 out_dim:int, kernel_size:int=3, stride:int=1, 
                 padding:int=1, dropout:float=0):
        
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = SwiGLU(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: spectrogram. Shape: B x 128 x 400
        x = self.linear(self.conv(x))
        x = self.activation(self.norm(x))
        return self.dropout(x)
    

def get_sinusoidal_pos_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: [seq_len, dim]