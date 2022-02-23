from pathlib import Path
import numpy as np
import pandas as pd
import python_speech_features as ps
from moviepy.editor import *
from tqdm import tqdm
import opensmile, argparse, pickle, pdb


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--data_dir', default='/media/data/public-data/SER')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    # save feature file
    save_feat_path = Path(args.data_dir).joinpath('federated_feature', args.feature_type)
    Path.mkdir(save_feat_path, parents=True, exist_ok=True)
    audio_features = {}

    if args.feature_type == 'emobase':
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.emobase,
                                feature_level=opensmile.FeatureLevel.Functionals)
    elif args.feature_type == 'ComParE':
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                feature_level=opensmile.FeatureLevel.Functionals)
    else:
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                feature_level=opensmile.FeatureLevel.Functionals)

    # msp-podcast
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

                audio_features[file_name] = {}
                audio_features[file_name]['data'] = np.array(smile.process_file(str(file_path)))
                audio_features[file_name]['session'] = session_id
                
    # crema-d
    elif args.dataset == 'crema-d':
        # data root folder
        data_root_path = Path(args.data_dir)
        file_list = [x for x in data_root_path.joinpath('AudioWAV').iterdir() if '.wav' in x.parts[-1]]
        file_list.sort()

        for file_path in tqdm(file_list, ncols=100, miniters=100):
            print('process %s' % file_path)
            if '1076_MTI_SAD_XX.wav' in str(file_path): continue
            file_name = file_path.parts[-1].split('.wav')[0]
            audio_features[file_name] = {}
            audio_features[file_name]['data'] = np.array(smile.process_file(str(file_path)))
            
    # iemocap
    elif args.dataset == 'iemocap':
        # data root folder
        session_list = [x.parts[-1] for x in Path(args.data_dir).iterdir() if  'Session' in x.parts[-1]]
        session_list.sort()
        for session_id in session_list:
            file_path_list = list(Path(args.data_dir).joinpath(session_id, 'sentences', 'wav').glob('**/*.wav'))
            for file_path in tqdm(file_path_list, ncols=100, miniters=100):
                file_name = file_path.parts[-1].split('.wav')[0].split('/')[-1]
                audio_features[file_name] = {}
                audio_features[file_name]['data'] = np.array(smile.process_file(str(file_path)))
    
    Path.mkdir(save_feat_path.joinpath(args.dataset), parents=True, exist_ok=True)
    save_path = str(save_feat_path.joinpath(args.dataset, 'data.pkl'))
    with open(save_path, 'wb') as handle:
        pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
            

