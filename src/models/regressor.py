import torch 
import torch.nn as nn

from modules import SwiGLU, ConvBlock, FeedForward
from config.regressor import ModelConfig



class BasicCNN(nn.Module):
    def __init__(self, last_dim:int, channels:list[int]=[128,256,64,32],
                 dims:list[int]=[400,256,128], dropout:float=0):
        super().__init__()
        dims.append(last_dim)
        assert(len(channels) == len(dims))

        depth = len(dims)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                in_channels=channels[i-1],
                out_channels=channels[i],
                in_dim=dims[i-1],
                out_dim=dims[i],
                dropout=dropout,
                kernel_size=3,
                stride=1,
                padding=1
            )
            for i in range(1,depth)
        ])

        self.final_ff = FeedForward(in_dim=dims[-1], mid_dim=dims[-1]*2, activation=SwiGLU(dims[-1]*2))


    def forward(self, x):
        # x: spectrogram. Shape: B x 128 x 400
        # returns tensor of shape: B x last_dim
        # B x T x F
        for block in self.conv_blocks:
            x = block(x)
        # B x T x F
        x = x.mean(dim = -1)

        return self.final_ff(x)
    


class LstmCNN(nn.Module):
    def __init__(self, base_model: BasicCNN, config: ModelConfig):
        super().__init__()

        self.cnn = base_model

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.audio_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.norm = nn.LayerNorm(config.audio_dim)

        self.audio_dropout = nn.Dropout(config.audio_dropout)

        self.time_embed = nn.Sequential(
            nn.Linear(1, config.time_dim*2),
            nn.LayerNorm(config.time_dim*2),
            SwiGLU(config.time_dim*2),
            nn.Linear(config.time_dim*2, config.time_dim),
            nn.LayerNorm(config.time_dim),
            nn.Dropout(config.time_dropout))
        
        self.time_scalar = nn.Parameter(torch.randn(1,  config.time_dim))

        hidden_concat = config.audio_dim + config.time_dim

        self.feedforward = FeedForward(in_dim=hidden_concat, mid_dim=hidden_concat*2, activation=SwiGLU(hidden_concat*2))
        self.final_norm = nn.LayerNorm(hidden_concat)

        self.class_head = nn.Linear(hidden_concat, 1)


    def forward(self, wav, emb_time):
        # x is a sequence of mel specs
        # x shape: B x seq_len x n_mels x num_frames
        # we process B x seq_len all together
        B, seq_len, n_fft, n_frames = wav.shape
        wav = wav.reshape(B*seq_len, n_fft, n_frames)
       
        cnn_res = self.cnn(wav)

        # B * seq_len x input_size -> B x seq_len x input_size
        cnn_res = cnn_res.reshape(B, seq_len, self.input_size)

        
        output, (hn, cn) = self.lstm(cnn_res)
        f_audio = self.norm(output.mean(dim=1) + hn.permute(1,0,2)[:, -1])
        f_audio = self.audio_dropout(f_audio)

        f_time = self.time_embed(emb_time) + self.time_scalar

        inputs_all = torch.cat([f_time, f_audio], dim=1)

        res = self.final_norm(self.feedforward(inputs_all))

        return self.class_head(res).squeeze(1)

