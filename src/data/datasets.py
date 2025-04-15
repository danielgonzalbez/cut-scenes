from torch.utils.data import Dataset
import torch
import torchaudio
import config

class AudioDataset(Dataset):
    def __init__(self, data, feature_extractor, normalize=True, dtype=torch.float32,
                 mean=None, std=None, tmask=None, fmask=None, noise=False):
        self.data = data
        self.feature_extractor=feature_extractor # not needed (just in case we want to compute the mel spec during training)
        self.normalize = normalize
        self.dtype=dtype
        self.mean = mean
        self.std=std
        self.noise = noise
        self.freqm = None
        self.timem = None

        if fmask:
            self.freqm = torchaudio.transforms.FrequencyMasking(fmask)
        if tmask:
            self.timem = torchaudio.transforms.TimeMasking(tmask)

        torch.set_default_dtype(torch.bfloat16)

    def __getitem__(self, idx):
        item = self.data[idx]

        mel = item['mel']

        if self.normalize:
            mel = (mel - self.mean)/(2*self.std)

        if self.freqm:
            mel = self.freqm(mel)
        if self.timem:
            mel = self.timem(mel)
        if self.noise:
            mel = mel + torch.rand(mel.shape[0], mel.shape[1]) * np.random.rand() / 10

        target = torch.tensor(item['label'], dtype=self.dtype)
        if not self.normalize:
          # predict samples
          target *= input_window
        if target > 1:
            target = torch.tensor(-1, dtype=self.dtype)

        time_tag = torch.tensor(item['empty_tag']).to(self.dtype)

        return mel, item['tag'], time_tag

    def __len__(self):
        return len(self.data)