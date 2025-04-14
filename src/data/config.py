sample_rate = 16000 # sample rate

# spectrogram params
window_time = 0.025
hop_time = 0.01

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

