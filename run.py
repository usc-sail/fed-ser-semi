import numpy as np
import os, pdb, sys
from pathlib import Path
import configparser


if __name__ == '__main__':

    # read config files
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')

    # 1. feature processing
    if config['mode'].getboolean('process_feature') is True:
        for dataset in ['iemocap', 'iemocap', 'crema-d']:
            if config['feature']['feature'] == 'emobase':
                cmd_str = 'taskset 100 python3 feature_extraction/opensmile_feature_extraction.py --dataset ' + dataset
            else:
                cmd_str = 'taskset 100 python3 feature_extraction/pretrained_audio_feature_extraction.py --dataset ' + dataset
            cmd_str += ' --feature_type ' + config['feature']['feature']
            cmd_str += ' --data_dir ' + config['dir'][dataset]
            cmd_str += ' --save_dir ' + config['dir']['save_dir']
            
            print('Extract features')
            print(cmd_str)
            pdb.set_trace()
            os.system(cmd_str)
    
    # 2. process training data
    if config['mode'].getboolean('process_training') is True:
        for dataset in ['msp-improv', 'iemocap', 'crema-d']:
            for feature in ['emobase', 'apc', 'vq_apc', 'tera', 'decoar2', 'npc']:
                cmd_str = 'taskset 100 python3 preprocess_data/preprocess_federate_data.py --dataset ' + dataset
                cmd_str += ' --feature_type ' + feature
                cmd_str += ' --data_dir ' + config['dir'][dataset]
                cmd_str += ' --save_dir ' + config['dir']['save_dir']
                cmd_str += ' --norm znorm'

                print('Process training data')
                print(cmd_str)
                os.system(cmd_str)

    # 3. Training SER model
    if config['mode'].getboolean('ser_training') is True:
        for dataset in ['iemocap', 'crema-d', 'msp-improv']:
            for feature in ['emobase', 'apc', 'tera', 'decoar2']:
                for client_label_rate in [0.1, 0.25]:
                    # cmd_str = 'taskset 1000 python3 train/federated_ser_classifier.py --dataset ' + dataset
                    cmd_str = 'taskset 1000 python3 train/federated_semi_ser_classifier.py --logit_threshold 0.5 --u 0.1 --dataset ' + dataset
                    cmd_str += ' --feature_type ' + feature
                    cmd_str += ' --dropout ' + config['model']['dropout']
                    cmd_str += ' --norm znorm --optimizer adam --client_label_rate ' + str(client_label_rate)
                    cmd_str += ' --model_type ' + config['model']['fed_model']
                    cmd_str += ' --learning_rate ' + config[config['model']['fed_model']]['lr']
                    cmd_str += ' --local_epochs ' + config[config['model']['fed_model']]['local_epochs']
                    cmd_str += ' --num_epochs ' + config[config['model']['fed_model']]['global_epochs']
                    cmd_str += ' --save_dir ' + config['dir']['save_dir']
                    
                    print('Traing SER model')
                    print(cmd_str)
                    os.system(cmd_str)
