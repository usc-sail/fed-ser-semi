from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import python_speech_features as ps
import pdb
from moviepy.editor import *
import argparse
import torchaudio
from tqdm import tqdm
import torch


def mfcc(audio):

    audio_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 1600, "hop_length": 160, "power": 2})
    mfcc = audio_transform(audio).detach()

    der1 = np.expand_dims(np.gradient(audio[0]), axis=0)
    der2 = np.expand_dims(np.gradient(audio[0], 2), axis=0)

    delta = audio_transform(torch.from_numpy(der1)).detach()
    ddelta = audio_transform(torch.from_numpy(der2)).detach()

    return np.concatenate((mfcc, delta, ddelta), axis=1)


def mel_spectrogram(audio, n_fft=1024, feature_len=128):

    window_size = n_fft
    window_hop = 160
    n_mels = feature_len
    window_fn = torch.hann_window

    audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=int(window_size),
        hop_length=int(window_hop),
        window_fn=window_fn,
    )

    audio_amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return audio_amp_to_db(audio_transform(audio).detach())


def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--feature_len', default=128)
    args = parser.parse_args()

    # save feature file
    root_path = Path('/media/data/projects/speech-privacy')
    create_folder(root_path.joinpath('federated_feature'))
    create_folder(root_path.joinpath('federated_feature', args.feature_type))
    create_folder(root_path.joinpath('federated_feature', args.feature_type, args.feature_len))
    save_feat_path = root_path.joinpath('federated_feature', args.feature_type, args.feature_len)
    
    audio_features = {}

    # msp-podcast
    if args.dataset == 'msp-podcast':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        data_str = 'MSP-podcast'
        label_df = pd.read_csv(data_root_path.joinpath(data_str, 'Labels', 'labels_concensus.csv'), index_col=0)
        
        audio_features['train'] = {}
        audio_features['validate'] = {}
        audio_features['test'] = {}
        
        for file_name in tqdm(list(label_df.index), ncols=100, miniters=100):
            file_path = data_root_path.joinpath(data_str, 'Audios', file_name)
            if Path.exists(file_path) is False:
                continue

            speaker_id = label_df.loc[file_name, 'SpkrID']
            gender = label_df.loc[file_name, 'Gender']

            if 'Test2' in label_df.loc[file_name, 'Split_Set']: continue
            if 'Unknown' in speaker_id or 'Unknown' in gender: continue

            audio, sample_rate = torchaudio.load(str(file_path))
            transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = transform_model(audio)
        
            audio_features[file_name] = {}

    # msp-improv
    elif args.dataset == 'msp-improv':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        session_list = [x.parts[-1] for x in data_root_path.joinpath('MSP-IMPROV', 'MSP-IMPROV', 'Audio').iterdir() if 'session' in x.parts[-1]]
        session_list.sort()
        
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath('MSP-IMPROV', 'MSP-IMPROV', 'Audio', session_id).glob('**/**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=50, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                print("process %s %s" % (session_id, file_name))

                audio, sample_rate = torchaudio.load(str(file_path))
                transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = transform_model(audio)
            
                audio_features[file_name] = {}
                if args.feature_type == 'mfcc':
                    audio_features[file_name]['data'] = mfcc(audio)
                else:
                    audio_features[file_name]['data'] = mel_spectrogram(audio, n_fft=400)
                audio_features[file_name]['session'] = session_id
                
    # crema-d
    elif args.dataset == 'crema-d':
        # data root folder
        data_root_path = Path('/media/data').joinpath('public-data', 'SER')
        file_list = [x for x in data_root_path.joinpath(args.dataset).iterdir() if '.wav' in x.parts[-1]]
        file_list.sort()

        for file_path in tqdm(file_list, ncols=100, miniters=100):
            print('process %s' % file_path)
            if '1076_MTI_SAD_XX.wav' in str(file_path):
                continue
            file_name = file_path.parts[-1].split('.wav')[0]
            audio, sample_rate = torchaudio.load(str(file_path))

            audio_features[file_name] = {}
            if args.feature_type == 'mfcc':
                audio_features[file_name]['data'] = mfcc(audio)
            else:
                audio_features[file_name]['data'] = mel_spectrogram(audio, n_fft=400)
            
    # iemocap
    elif args.dataset == 'iemocap':
        # data root folder
        data_root_path = Path('/media/data').joinpath('sail-data')
        session_list = [x.parts[-1] for x in data_root_path.joinpath(args.dataset).iterdir() if  'Session' in x.parts[-1]]
        session_list.sort()
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath(args.dataset, session_id, 'sentences', 'wav').glob('**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=100, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                audio, sample_rate = torchaudio.load(str(file_path))

                audio_features[file_name] = {}
                if args.feature_type == 'mfcc':
                    audio_features[file_name]['data'] = mfcc(audio)
                else:
                    audio_features[file_name]['data'] = mel_spectrogram(audio, n_fft=400)
                
    create_folder(save_feat_path.joinpath(args.dataset))
    save_path = str(save_feat_path.joinpath(args.dataset, 'data.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
            

