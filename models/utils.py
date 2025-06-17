import torch
import math
import torch.nn.init as init
import torch.nn as nn


def get_sinusoidal_pos_encoding(seq_len, dim):
    pe = torch.zeros(seq_len, dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: [seq_len, dim]


def initialize_weights(model, bias_class_head=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  

    nn.init.zeros_(model.time_scalar)

    if bias_class_head:
        model.class_head.bias.data.fill_(bias_class_head)
