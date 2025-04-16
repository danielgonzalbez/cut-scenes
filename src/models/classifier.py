import torch.nn as nn
import torch 

from modules import SwiGLU, MultiHeadAttention, FeedForward
from config.classifier import CNNConfig, ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, in_dim:int, out_dim:int, kernel_size:int=3,
                 stride:int=1, padding:int=1, dropout:float=0):
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



class BasicCNN(nn.Module):
    def __init__(self, last_dim:int, channels:list[int]=[128,256,64,32],
                 dims:list[int]=[400,256,128], dropout:float=0):
        super().__init__()
        dims.append(last_dim)
        assert(len(channels) == len(dims))
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                in_channels=channels[i-1],
                out_channels=channels[i],
                in_dim=dims[i-1],
                out_dim=dims[i],
                dropout=dropout
            )
            for i in range(1, len(channels))
        ])

        self.final_conv = nn.Conv1d(in_channels=channels[-1], out_channels=1, kernel_size=3, stride=1, padding=1)
        self.final_norm = nn.LayerNorm(last_dim)

    def forward(self, x):
        # x: spectrogram. Shape: B x 128 x 400
        # returns tensor of shape: B x last_dim
        for block in self.conv_blocks:
            x = block(x)
        x = self.final_conv(x).squeeze(1)
        x = self.final_norm(x)
        return x


class BasicModel(nn.Module):
    def __init__(self, base_model: BasicCNN, inner_dim:int=16,
                 depth_attn:int=1, num_heads:int=1, head_dim:int=None,
                 drop_prev_samples:float=0, drop_proj:float=0):

        super().__init__()
        self.base = base_model

        if not head_dim:
           head_dim = inner_dim

        self.swiglu = SwiGLU(inner_dim)

        self.emb_previous_samples = nn.Sequential(
            nn.Linear(1, inner_dim),
            nn.LayerNorm(inner_dim),
            self.swiglu,
            nn.Linear(inner_dim,inner_dim),
            nn.LayerNorm(inner_dim),
            nn.Dropout(drop_prev_samples)
        )

        self.attn_blocks = nn.ModuleList(
           MultiHeadAttention(in_dim=1, num_heads=num_heads, head_dim=head_dim)
         for _ in range(depth_attn))

        self.ff_blocks = nn.ModuleList(
           FeedForward(in_dim=inner_dim*2, mid_dim=inner_dim*4, activation=nn.SiLU())
         for _ in range(depth_attn))

        self.norm_layers = nn.ModuleList(
            nn.LayerNorm(inner_dim*2)
         for _ in range(depth_attn))

        self.drop_proj = nn.Dropout(drop_proj)

        self.final_layer = nn.Linear(inner_dim*2, 1)


    def forward(self, x, prev_samples_idx):
        # x shape: B, H, W
        x = self.base(x).squeeze(1) # B x inner_dim

        emb = self.emb_previous_samples(prev_samples_idx) # B x inner_dim

        features = torch.cat([x, emb], dim=1)

        for attn, ff, norm in zip(self.attn_blocks, self.ff_blocks, self.norm_layers):
            skip = features
            features = attn(features.unsqueeze(2)).squeeze(2)
            features = ff(features) + skip
            features = self.drop_proj(norm(features))


        out = self.final_layer(features)

        return out.squeeze(1)
