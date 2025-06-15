import os
from tqdm import tqdm
import json
import torchaudio
import torch
import numpy as np

import config
from feature_extractor import FeatureExtractor

folders = [
    subf for subf in os.listdir(config.folder)
    if os.path.isdir(os.path.join(config.folder, subf))
]


def generate_overlapping_samples(targets_wav:str, input_window:int = config.input_window, 
                                 sliding_hop:int = config.sliding_hop, name:str = None) -> list[dict]:
    """
    Returns a list of dictionaries with the properties needed to identify an audio window:
        - start: first sample of the audio window
        - end: last sample of the audio window
        - empty_samples: normalized number of samples since the last shot change
        - label: normalized number of samples until the next shot change
        - source: id of the processed wav file
    """
    idx_target = 1
    current_sample = 0
    windows = []
    while idx_target < len(targets_wav):
        assert(targets_wav[idx_target]>current_sample)
        label = (targets_wav[idx_target] - current_sample) / input_window
        empty_samples = (current_sample - targets_wav[idx_target-1]) / input_window
        windows.append({
                "start": current_sample,
                "end": min(current_sample + input_window, targets_wav[-1]),
                "empty_samples": empty_samples,
                "label": label,
                "source": name
            })
        while idx_target < len(targets_wav) and current_sample + sliding_hop >= targets_wav[idx_target]:
            idx_target += 1
        current_sample += sliding_hop
    return windows



def split_train_test_eval_data(data: list[dict], eval_split: tuple, test_folders: list[str], lengths_wavs: dict):
    test_data = [d for d in data if d['source'].strip() in test_folders]

    eval_data = [d for d in data if d['end'] < 0.65 * lengths_wavs[d['source']] 
                 and d['start'] > 0.5* lengths_wavs[d['source']] 
                 and d['source'].strip() not in test_folders]

    train_data = [d for i,d in enumerate(data) if 
                  (d['end'] < 0.5 * lengths_wavs[d['source']] or d['start'] > 0.65*lengths_wavs[d['source']]) 
                and d['source'].strip() not in test_folders]

    return train_data, eval_data, test_data



def clean_samples(sample: dict, special_cases: dict, lengths_wavs: dict):
    name = sample['source']

    if sample['start'] < config.start_sample or sample['end'] > lengths_wavs[name] - config.end_sample:
        return False

    if name in special_cases:
        if 'end' in special_cases[name] and sample['end'] > lengths_wavs[name] - special_cases[name]['end']:
            return False
        if 'start' in special_cases[name] and sample['start'] < special_cases[name]['start']:
            return False
    return True


def compute_mels(feature_extractor: FeatureExtractor, data:list[dict]):
    """Extracts mels from each sample and returns the mean and std deviation for later use"""
    means = []
    stds = []
    # extract mel from each sample
    for subfolder in tqdm(folders):
        files = os.listdir(os.path.join(config.folder, subfolder))
        wav_file = [f for f in files if f.endswith('.wav')][0]
        re_wav, _ = torchaudio.load(os.path.join(config.folder,subfolder,wav_file), normalize=True)
        i_subfolder = [i for i, d in enumerate(data) if d['source'].strip()==subfolder] 
        for i in i_subfolder:
            sample = data[i]
            wav_left = re_wav[0]
            wav_left = wav_left.squeeze()[round(sample['start']):round(sample['end']-1)]
            wav_right = re_wav[1]
            wav_right = wav_right.squeeze()[round(sample['start']):round(sample['end']-1)]
            wav = (wav_left + wav_right)/2
            wav = wav - wav.mean() # DC offset
            mel = feature_extractor(wav)
            assert(mel.shape[-1]==config.num_frames)
            data[i]['mel'] = mel.to(torch.bfloat16)
            means.append(mel.mean().item())
            stds.append(mel.std().item())
            del mel

    return data, np.mean(means), np.mean(stds)



def generate_data(eval_split: tuple=config.eval_split, 
                  test_folders: list[str]=config.test_folders,
                  precompute_mels: bool=config.precompute_mels):
    """
    Generates the samples of train, eval and test sets and filters noisy samples.
    Args:
        - eval_split: tuple indicating the start and end of the audio segments corresponding to the eval split.
        - test_folders: list of video ids of the test split
    Returns: 
        - 3 lists of dictionaries (train, eval and test).
    """
    lengths_wavs = {} # we save the length of each audio in case we need it later
    for subfolder in tqdm(folders):
        files = os.listdir(os.path.join(config.folder, subfolder))
        wav_file = [f for f in files if f.endswith('.wav')][0]
        target_file = os.path.join(config.folder, subfolder, 'timestamps/metadata.json')
        re_wav, sample_rate = torchaudio.load(os.path.join(config.folder,subfolder,wav_file), normalize=True)
        lengths_wavs[subfolder.strip()] = len(re_wav[0])
        with open(target_file) as file:
            json_file = json.load(file)
        targets = json_file['FINAL_TIMESTAMPS']
        targets_wav = [int(t[0] * sample_rate) for t in targets] + [int(targets[-1][1] * sample_rate)]
        data += generate_overlapping_samples(re_wav, targets_wav, input_window=config.input_window,
                                                sliding_hop=config.sliding_hop, name=subfolder.strip())

    # handle special cases 
    with open(config.special_cases_file) as f:
        special_cases = json.load(f)
    data = [d for d in data if clean_samples(d, special_cases, lengths_wavs)]

    # clen samples
    data = [d for d in data if d['empty_samples']+ d['label']*config.input_window < 16000*50]
    data = [d for d in data if d['label']*config.input_window > 4.2*16000 or d['label']*config.input_window < 3.8*16000]
    data = [d for d in data if d['start'] > 150*16000 and d['end'] < lengths_wavs[d['source'].strip()] - 350*16000]
    data = [d for d in data if d['label'] * config.input_window > 0.2*16000]
    data = [d for d in data if d['empty_samples'] > 0.2*16000]

    # split sets
    train_data, eval_data, test_data = split_train_test_eval_data(data, eval_split, test_folders, lengths_wavs)

    print(f"Generated {len(train_data)} train samples, {len(eval_data)} eval samples and {len(test_data)} test samples")

    # add mel specs to improve data loading efficiency
    if precompute_mels:
        feature_extractor = FeatureExtractor(normalize_input=False)
        train_data, mean_train, std_train = compute_mels(feature_extractor=feature_extractor, data=train_data)
        eval_data, _, _ = compute_mels(feature_extractor=feature_extractor, data=eval_data)
        test_data, _, _ = compute_mels(feature_extractor=feature_extractor, data=test_data)
    
    return train_data, eval_data, test_data, (mean_train, std_train)








