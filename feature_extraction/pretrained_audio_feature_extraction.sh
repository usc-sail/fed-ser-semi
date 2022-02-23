# This is the script for opensmile feature extraction

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# features include wav2vec

# apc, tera, decoar2, audio_albert, cpc, distilhubert, mockingjay, wav2vec2, npc

taskset 100 python3 pretrained_audio_feature_extraction.py --dataset msp-improv --feature_type vq_wav2vec
taskset 100 python3 pretrained_audio_feature_extraction.py --dataset iemocap --feature_type vq_wav2vec
taskset 100 python3 pretrained_audio_feature_extraction.py --dataset crema-d --feature_type vq_wav2vec
