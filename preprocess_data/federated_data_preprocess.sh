# This is the script for training data preprocess
# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# cpc, apc, tera, decoar2, audio_albert, distilhubert

taskset 100 python3 preprocess_federate_data.py --dataset msp-improv \
                        --feature_type emobase --norm znorm
                        --data_dir /media/data/sail-data/MSP-IMPROV/MSP-IMPROV 
                        --save_dir /media/data/projects/speech-privacy 