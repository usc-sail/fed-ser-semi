from pathlib import Path
import numpy as np
import pandas as pd
from moviepy.editor import *
import argparse, pdb, pickle
import torchaudio, torch
from tqdm import tqdm
import s3prl.hub as hub

sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

def pretrained_feature(audio):
    
    save_feature = []
    if args.feature_type == 'wav2vec':
        with torch.inference_mode(): 
            features, _ = model.extract_features(audio)
        for idx in range(len(features)): save_feature.append(np.mean(features[idx].detach().cpu().numpy(), axis=1))
    elif args.feature_type == 'distilhubert' or args.feature_type == 'wav2vec2' or args.feature_type == 'vq_wav2vec' or args.feature_type == 'cpc':
        features = model([audio[0]])['last_hidden_state']
    else:
        features = model(audio)['last_hidden_state']
    save_feature.append(np.mean(features.detach().cpu().numpy(), axis=1))
    save_feature.append(np.std(features.detach().cpu().numpy(), axis=1))
    save_feature.append(np.max(features.detach().cpu().numpy(), axis=1))
    del features
    return save_feature

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='wav2vec')
    parser.add_argument('--data_dir', default='/media/data/public-data/SER')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    # save feature file
    save_feat_path = Path(args.save_dir).joinpath('federated_feature', args.feature_type)
    Path.mkdir(save_feat_path, parents=True, exist_ok=True)
    audio_features = {}

    # Model related
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    if args.feature_type == 'wav2vec':
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
    elif args.feature_type == args.feature_type:
        model = getattr(hub, args.feature_type)()
    model = model.to(device)
  
    # msp-improv
    if args.dataset == 'msp-improv':
        # data root folder
        data_root_path = Path(args.data_dir)
        session_list = [x.parts[-1] for x in data_root_path.joinpath('Audio').iterdir() if 'session' in x.parts[-1]]
        session_list.sort()
        
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath('Audio', session_id).glob('**/**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=50, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                print("process %s %s" % (session_id, file_name))

                audio, sample_rate = torchaudio.load(str(file_path))
                transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = transform_model(audio)
                audio = audio.to(device)
            
                audio_features[file_name] = {}
                save_feature = pretrained_feature(audio)
                audio_features[file_name]['data'] = save_feature
                audio_features[file_name]['session'] = session_id
                del save_feature

    # crema-d
    elif args.dataset == 'crema-d':
        # data root folder
        data_root_path = Path(args.data_dir)
        file_list = [x for x in data_root_path.joinpath('AudioWAV').iterdir() if '.wav' in x.parts[-1]]
        file_list.sort()

        for file_path in tqdm(file_list, ncols=100, miniters=100):
            print('process %s' % file_path)
            if '1076_MTI_SAD_XX.wav' in str(file_path):
                continue
            file_name = file_path.parts[-1].split('.wav')[0]
            audio, sample_rate = torchaudio.load(str(file_path))
            audio = audio.to(device)

            audio_features[file_name] = {}
            save_feature = pretrained_feature(audio)
            audio_features[file_name]['data'] = save_feature
            del save_feature
            
    # iemocap
    elif args.dataset == 'iemocap':
        # data root folder
        data_root_path = Path(args.data_dir)
        session_list = [x.parts[-1] for x in data_root_path.iterdir() if 'Session' in x.parts[-1]]
        session_list.sort()
        for session_id in session_list:
            file_path_list = list(data_root_path.joinpath(session_id, 'sentences', 'wav').glob('**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=100, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                audio, sample_rate = torchaudio.load(str(file_path))
                audio = audio.to(device)

                audio_features[file_name] = {}
                save_feature = pretrained_feature(audio)
                audio_features[file_name]['data'] = save_feature
                del save_feature

    Path.mkdir(save_feat_path.joinpath(args.dataset), parents=True, exist_ok=True)
    save_path = str(save_feat_path.joinpath(args.dataset, 'data.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
