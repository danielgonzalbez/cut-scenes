import torch.nn as nn
import torch 

from .modules import SwiGLU, MultiHeadAttention, FeedForward
from .utils import get_sinusoidal_pos_encoding
from .config.classifier import ModelConfig


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

        self.final_linear = nn.Sequential(
                    nn.Linear(dims[-1], dims[-1]*2),
                    nn.LayerNorm(dims[-1]*2),
                    nn.SiLU(),
                    nn.Linear(dims[-1]*2, dims[-1])
                )        
        self.final_norm = nn.LayerNorm(last_dim)


    def forward(self, x):
        # x: spectrogram. Shape: B x 128 x 400
        # returns tensor of shape: B x last_dim
        for block in self.conv_blocks:
            x = block(x)
        x = self.final_linear(x)
        x = self.final_norm(x)
        return x


class BasicModel(nn.Module):
    def __init__(self, base_model: BasicCNN, config: ModelConfig):

        super().__init__()
        self.base = base_model

        self.emb_previous_samples = nn.Sequential(
                nn.Linear(1, config.time_dim),
                nn.LayerNorm(config.time_dim),
                SwiGLU(config.time_dim),
                nn.Dropout(config.time_dropout),
                nn.Linear(config.time_dim,config.time_dim),
                nn.LayerNorm(config.time_dim),
                nn.Dropout(config.time_dropout)
            )

        self.attn_blocks = nn.ModuleList(
           MultiHeadAttention(in_dim=config.audio_dim, num_heads=config.num_heads, head_dim=config.head_dim)
         for _ in range(config.depth_attn))

        self.ff_blocks = nn.ModuleList(
           FeedForward(in_dim=config.audio_dim, mid_dim=config.audio_dim, activation=SwiGLU(config.audio_dim))
         for _ in range(config.depth_attn))

        self.norm_layers = nn.ModuleList(
            nn.LayerNorm(config.audio_dim)
         for _ in range(config.depth_attn-1))


        self.ff_audio = nn.Sequential(
                    nn.Linear(config.audio_dim*config.seq_len, config.audio_dim*2),
                    nn.LayerNorm(config.audio_dim*2),
                    SwiGLU(config.audio_dim*2),
                    nn.Dropout(config.audio_dropout),
                    nn.Linear(config.audio_dim*2,config.audio_dim),
                    nn.LayerNorm(config.audio_dim),
                    nn.Dropout(config.audio_dropout)
                )
        
        self.flatten = nn.Flatten(start_dim=1)

        self.time_scalar = nn.Parameter(torch.randn(1,  config.time_dim))

        self.previous_class_head = nn.Sequential(
                                        nn.Linear(config.audio_dim + config.time_dim, config.audio_dim),
                                        nn.LayerNorm(config.audio_dim),
                                        nn.SiLU())
        
        self.class_head = nn.Linear(config.audio_dim,1)

        self.final_dropout = nn.Dropout(config.final_dropout)
            


    def forward(self, x, prev_samples):
        # x shape: B, H, W
        x = self.base(x).squeeze(1) # B x inner_dim

        x = x.permute(0,2,1) # B x T x F (to treat time in the sequence dimension)
        B, T, F = x.shape

        pos_encoding = get_sinusoidal_pos_encoding(T, F).unsqueeze(0)  # [1, T, F]
        f_audio = x + pos_encoding.to(x.device)  # [B, T, F]

        for i, (attn, ff) in enumerate(zip(self.attn_blocks, self.ff_blocks)):
            attn_features = attn(f_audio)
            new_features = ff(attn_features) + f_audio
            if i < len(self.attn_blocks) - 1: 
                f_audio = self.norm_layers[i](new_features)

        f_audio = self.flatten(f_audio)
        f_audio = self.ff_audio(f_audio)
            
        f_time = self.emb_previous_samples(prev_samples) + self.time_scalar # B x inner_dim
        
        f_total = torch.cat([f_audio, f_time], dim=1)

        out = self.previous_class_head(self.final_dropout(f_total))
        out = self.class_head(out)

        return out.squeeze(1)#, self.final_layer_emb(emb).squeeze(1), self.final_layer_mel(x).squeeze(1)


