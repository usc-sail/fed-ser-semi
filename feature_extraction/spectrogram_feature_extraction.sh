# This is the script for opensmile feature extraction

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# features include mel spectrogram and mfcc

taskset 100 python3 spectrogram_feature_extraction.py --dataset msp-improv --feature_type mel_spec --feature_len 80
taskset 100 python3 spectrogram_feature_extraction.py --dataset iemocap --feature_type mel_spec --feature_len 80
taskset 100 python3 spectrogram_feature_extraction.py --dataset crema-d --feature_type mel_spec --feature_len 80
