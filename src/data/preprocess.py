import config
import os
from tqdm import tqdm
import json
import torchaudio

folders = [
    subf for subf in os.listdir(config.folder)
    if os.path.isdir(os.path.join(config.folder, subf))
]


lengths_wavs = {} # we save the length of each audio in case we need it later

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


def clean_samples(sample: dict, special_cases: dict):
    name = sample['source']

    if sample['start'] < config.start_sample or sample['end'] > lengths_wavs[name] - config.end_sample:
        return False

    if name in special_cases:
        if 'end' in special_cases[name] and sample['end'] > lengths_wavs[name] - special_cases[name]['end']:
            return False
        if 'start' in special_cases[name] and sample['start'] < special_cases[name]['start']:
            return False
        
    return True



def generate_dataset():
    """
    Generates the samples of train, eval and test sets and filters noisy samples.
    Returns a list of dictionaries.
    """
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
    data = [d for d in data if clean_samples(d, special_cases)]

    return data




