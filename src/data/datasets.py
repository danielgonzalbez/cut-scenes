
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, data, feature_extractor, normalize=True, dtype=torch.float32,
                 mean=None, std=None, tmask=None, fmask=None, noise=False,
                 mean_empty=None, std_empty=None):
        self.data = data
        self.feature_extractor=feature_extractor
        self.normalize = normalize
        self.dtype=dtype
        self.mean = mean
        self.std=std
        self.noise = noise
        self.freqm = fmask 
        self.timem = tmask
        self.mean_empty = mean_empty
        self.std_empty = std_empty
        
        if fmask:
            self.freqm = torchaudio.transforms.FrequencyMasking(fmask)
        if tmask:
            self.timem = torchaudio.transforms.TimeMasking(tmask)
        torch.set_default_dtype(self.dtype)

    def compute_frame(self,start_sample):
        secs = start_sample/16000 #.item()
        frame_num = math.floor(secs * 5) - 1
        zeros = '0'*(4-len(str(frame_num)))
        return f"{zeros}{frame_num}"

    def compute_embs_images(self, source, start_sample):
        frame = self.compute_frame(start_sample)
        image = Image.open(f"/kaggle/input/concert-frames/{source}/frame_{frame}.jpg")
        if self.noise:
          return self._train_transforms(image)
        else:
          return self._val_transforms(image)

    def __getitem__(self, idx):
        item = self.data[idx]

        mel = item['mel']
        mel = (mel - self.mean)/(2*self.std)

        if self.freqm:
            mel = self.freqm(mel)
        if self.timem:
            mel = self.timem(mel)
        if self.noise:
            mel = mel + torch.rand(mel.shape[0], mel.shape[1]) * np.random.rand() / 10        
        
        time_info = torch.tensor((item['empty_samples']-self.mean_empty)/(self.std_empty)).to(self.dtype) 
        if self.noise:
            jitter_std = 0.02
            time_info += torch.randn_like(time_info) * jitter_std

        return mel.to(self.dtype), time_info, item['label'], torch.tensor(item['empty_samples']), item['source'] 

    def __len__(self):
        return len(self.data)