sample_rate = 16000 # sample rate

# spectrogram params
window_time = 0.025
hop_time = 0.01
n_mels = 128
n_fft = 512

# training params:
folder = "../data/concerts-scenes"
secs = 4 # number of seconds in the processed audio
num_frames = secs/hop_time # number of windows in the processed audio
input_window = int(secs*sample_rate) # number of samples in the processed audio
factor = 5 
sliding_hop = round(input_window / factor) # number of samples without overlap between consecutive windows

# dataset
special_cases_file = "special_cases.json" # file with the special audio files to handle
start_sample = 150 * sample_rate # first sample to process
end_sample = 350 * sample_rate # length(audio) - end_sample will be the last sample to process
test_folders = ['-oxH-7VklBI', 'd3gh9l37Yt8', 'oi6Vn8WxFIc','JrOJK5kf7eM','1Vm9RMR80D0', '40xhyXscddY', 'LDBmGj9xxpM',
                '6exoB7IW8qw', '5SNxIjdjB1o', '8FP8Jf2gFrM', '3tisvEpblig', '1xXsx3ggR5Y', 'AgXW-57UDMc', 'NjIduF3equQ',
               'KRUP0TgBN3E']
precompute_mels = True # whether we want to precompute the mel specs to improve data loading efficiency during training
tmask = 128 # only applied in training
fmask = 48 # only applied in training
noise=True # only applied in training