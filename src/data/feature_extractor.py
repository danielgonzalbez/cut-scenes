import torchaudio
import torch
from .config import window_time, hop_time, n_fft, n_mels, sample_rate

# Log mel spec extractor:

class FeatureExtractor():
    def __init__(self, normalize_input=False, mean=-4.2677393,std=4.5689974, window_time=window_time,
                 hop_time=hop_time, sample_rate=sample_rate, dtype=torch.float32, 
                 n_mels=n_mels, n_fft=n_fft):
        torch.set_default_dtype(dtype)
        hop_length = int(hop_time * sample_rate) # default: 160
        win_length = int(window_time * sample_rate) # default: 400

        self.transforms_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                   win_length=win_length,
                                                          hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)
        self.mean = mean
        self.std = std
        self.normalize_input = normalize_input
        self.dtype = dtype


    def __call__(self, audio):
        mel = self.transforms_mel(audio).to(self.dtype)
        # normalization following: https://github.com/YuanGongND/ast/blob/102f0477099f83e04f6f2b30a498464b78bbaf46/src/dataloader.py#L191
        mel = torch.log(torch.clamp(mel,1e-10))
        if self.normalize_input:
            assert(self.mean and self.std)
            return self.normalize(mel, self.mean, self.std)
        return mel

    def normalize(self, audio, mean, std):
        return (audio - mean) / (2*std)


