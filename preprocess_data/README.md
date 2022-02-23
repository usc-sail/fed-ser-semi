# Preprocess features
The scripts under this folder prepares the ready-to-go data (in pickle files) for training. 

### Quick example
The bash file federated_data_preprocess.sh provides an example of running the preprocess python file. e.g.:

```sh
taskset 100 python3 preprocess_federate_data.py --dataset msp-improv \
                        --feature_type emobase --norm znorm
                        --data_dir /media/data/sail-data/MSP-IMPROV/MSP-IMPROV 
                        --save_dir /media/data/projects/speech-privacy
```
- The arg `dataset` specifies the data set. The support data sets are IEMOCAP, MSP-Improv, and CREMA-D. 

- The arg `feature_type` is the feature reprentation type. Please refer to README under feature extraction for more details.

- The arg `pred` is prediction label. Currently support SER only, the arousal and valence predictions are ongoing work.

- The arg `norm` specifies the normalization method including z-normalization and min-max normalization. The normalization is implemented within a speaker.

- The arg `data_dir` specifies the raw data location, we use it to read the groundtruth labels.

- The arg `save_dir` specifies the save location, we use it to save all the processed data.
