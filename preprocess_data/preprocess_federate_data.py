from operator import sub
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle, argparse, re, pdb
from sklearn.model_selection import KFold

emo_map_dict = {'N': 'neu', 'S': 'sad', 'H': 'hap', 'A': 'ang'}
emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}

speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}
    
def save_data_dict(save_data, label, gender, speaker_id):
    if speaker_id in test_speaker_id_arr:
        if speaker_id not in test_dict: 
            test_dict[speaker_id] = {}
            test_dict[speaker_id]['data'] = []
            test_dict[speaker_id]['label'] = []
        test_dict[speaker_id]['data'].append(save_data.copy())
        test_dict[speaker_id]['label'].append(emo_dict[label])
    elif speaker_id in train_speaker_id_arr:
        if speaker_id not in training_dict: 
            training_dict[speaker_id] = {}
            training_dict[speaker_id]['data'] = []
            training_dict[speaker_id]['label'] = []
        
        training_dict[speaker_id]['data'].append(save_data.copy())
        training_dict[speaker_id]['label'].append(emo_dict[label])
        
def save_non_iid(data_dict, type='non'):
    speaker_ids = list(data_dict.keys())
    for speaker_id in speaker_ids:
        idxs = np.arange(len(data_dict[speaker_id]['label']))
        idxs_labels = np.vstack((idxs, np.array(data_dict[speaker_id]['label'])))
        idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
        
        idxs, labels = idxs_labels[0], idxs_labels[1]
        class_dist = {}
        specific_class_dict = {}
        for class_idx in range(4):
            specific_class = np.extract(labels == class_idx, idxs)
            class_dist[class_idx] = len(specific_class)
            specific_class_dict[class_idx] = specific_class
        
        # we split each client's data to four shards, and each shard is with 3 labels
        if args.dataset != 'crema-d':
            num_sub_clients = 4
            for sub_client_idx in range(num_sub_clients):
                train_idx = []
                sub_client_data_dict = {}
                sub_client_data_dict['data'], sub_client_data_dict['label'] = [], []
                for class_idx in range(4):
                    # skip the class
                    if class_idx == sub_client_idx: continue
                    num_samples = len(specific_class_dict[class_idx]) if len(specific_class_dict[class_idx]) < int(class_dist[class_idx]/3) + 1 else int(class_dist[class_idx]/3) + 1
                    train_tmp = np.random.choice(specific_class_dict[class_idx], num_samples, replace=False)
                    train_idx = train_idx + list(train_tmp)
                    specific_class_dict[class_idx] = np.array(list(set(specific_class_dict[class_idx])-set(train_tmp)))
                
                # append data
                for idx in train_idx:
                    sub_client_data_dict['data'].append(data_dict[speaker_id]['data'][idx])
                    sub_client_data_dict['label'].append(data_dict[speaker_id]['label'][idx])
                
                # z-normalization per client
                speaker_mean = np.nanmean(np.array(sub_client_data_dict['data']), axis=0)
                speaker_std = np.nanstd(np.array(sub_client_data_dict['data']), axis=0)
                for idx in range(len(np.array(sub_client_data_dict['data']))):
                    sub_client_data_dict['data'][idx] = (sub_client_data_dict['data'][idx]-speaker_mean) / (speaker_std+1e-5)
                
                client_val_idx = np.random.choice(np.arange(len(train_idx)), int(len(train_idx)*0.2), replace=False)
                client_train_idx = list(set(np.arange(len(train_idx)))-set(client_val_idx))
                sub_client_train_dict, sub_client_val_dict = {}, {}
                
                sub_client_train_dict['data'] = np.array(sub_client_data_dict['data'])[client_train_idx]
                sub_client_train_dict['label'] = np.array(sub_client_data_dict['label'])[client_train_idx]
                sub_client_val_dict['data'] = np.array(sub_client_data_dict['data'])[client_val_idx]
                sub_client_val_dict['label'] = np.array(sub_client_data_dict['label'])[client_val_idx]
                
                # dump the data
                Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold, 'train'), parents=True, exist_ok=True)
                f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'train', str(speaker_id)+'_'+str(sub_client_idx)+'.pkl')), "wb")
                pickle.dump(sub_client_train_dict, f)
                f.close()
                
                Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold, 'validation'), parents=True, exist_ok=True)
                f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'validation', str(speaker_id)+'_'+str(sub_client_idx)+'.pkl')), "wb")
                pickle.dump(sub_client_val_dict, f)
                f.close()
        else:
            sub_client_data_dict = {}
            sub_client_data_dict['data'], sub_client_data_dict['label'] = [], []
            
            # append data
            for idx in range(len(data_dict[speaker_id]['label'])):
                sub_client_data_dict['data'].append(data_dict[speaker_id]['data'][idx])
                sub_client_data_dict['label'].append(data_dict[speaker_id]['label'][idx])
            
            # z-normalization per client
            speaker_mean = np.nanmean(np.array(sub_client_data_dict['data']), axis=0)
            speaker_std = np.nanstd(np.array(sub_client_data_dict['data']), axis=0)
            for idx in range(len(np.array(sub_client_data_dict['data']))):
                sub_client_data_dict['data'][idx] = (sub_client_data_dict['data'][idx]-speaker_mean) / (speaker_std+1e-5)
            
            data_len = len(sub_client_data_dict['data'])
            client_val_idx = np.random.choice(np.arange(data_len), int(data_len*0.2), replace=False)
            client_train_idx = list(set(np.arange(data_len))-set(client_val_idx))
            sub_client_train_dict, sub_client_val_dict = {}, {}
            
            sub_client_train_dict['data'] = np.array(sub_client_data_dict['data'])[client_train_idx]
            sub_client_train_dict['label'] = np.array(sub_client_data_dict['label'])[client_train_idx]
            sub_client_val_dict['data'] = np.array(sub_client_data_dict['data'])[client_val_idx]
            sub_client_val_dict['label'] = np.array(sub_client_data_dict['label'])[client_val_idx]
            
            # dump the data
            Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold, 'train'), parents=True, exist_ok=True)
            f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'train', str(speaker_id)+'.pkl')), "wb")
            pickle.dump(sub_client_train_dict, f)
            f.close()
            
            Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold, 'validation'), parents=True, exist_ok=True)
            f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'validation', str(speaker_id)+'.pkl')), "wb")
            pickle.dump(sub_client_val_dict, f)
            f.close()

def save_test(data_dict):
    speaker_ids = list(data_dict.keys())
    for speaker_id in speaker_ids:
        # z-normalization per client
        speaker_mean = np.nanmean(np.array(data_dict[speaker_id]['data']), axis=0)
        speaker_std = np.nanstd(np.array(data_dict[speaker_id]['data']), axis=0)
        for idx in range(len(np.array(data_dict[speaker_id]['data']))):
            data_dict[speaker_id]['data'][idx] = (data_dict[speaker_id]['data'][idx]-speaker_mean) / (speaker_std+1e-5)

        # dump the data
        Path.mkdir(preprocess_path.joinpath(data_set_str, test_fold, 'test'), parents=True, exist_ok=True)
        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'test', str(speaker_id)+'.pkl')), "wb")
        pickle.dump(data_dict[speaker_id], f)
        f.close()
        
        
if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--data_dir', default='/media/data/public-data/SER')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    # get the 5 different test folds
    speaker_id_arr = speaker_id_arr_dict[args.dataset]    
    train_array, test_array = [], []
    
    # read args
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    fold_idx, feature_type, data_set_str = 1, args.feature_type, args.dataset
    for train_index, test_index in kf.split(speaker_id_arr):
        
        # 80% are training (80% of data on a client is for training, rest validation), and 20% are test
        train_arr, test_arr = speaker_id_arr[train_index], speaker_id_arr[test_index]
        test_fold = 'fold'+str(fold_idx)
        print('Process %s training set with test %s' % (data_set_str, test_fold))
        
        # save preprocess file dir
        preprocess_path = Path(args.save_dir).joinpath('fed-ser-semi', feature_type, args.pred)
        Path.mkdir(preprocess_path, parents=True, exist_ok=True)

        # feature folder
        feature_path = Path(args.save_dir).joinpath('federated_feature', feature_type)
        training_norm_dict = {}

        # read features
        with open(feature_path.joinpath(data_set_str, 'data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        
        training_dict, test_dict = {}, {}
        if data_set_str == 'msp-improv':
            # data root folder
            sentence_file_list = list(data_dict.keys())
            sentence_file_list.sort()
            speaker_id_list = ['M01', 'F01', 'M02', 'F02', 'M03', 'F03', 'M04', 'F04', 'M05', 'F05', 'M06', 'F06']

            train_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in train_arr]
            test_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)
            
            # data root folder
            evaluation_path = Path(args.data_dir).joinpath('Evalution.txt')
            with open(str(evaluation_path)) as f:
                evaluation_lines = f.readlines()

            label_dict = {}
            for evaluation_line in evaluation_lines:
                if 'UTD-' in evaluation_line:
                    file_name = 'MSP-'+evaluation_line.split('.avi')[0][4:]
                    label_dict[file_name] = evaluation_line.split('; ')[1][0]
                    
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_part = sentence_file.split('-')
                recording_type = sentence_part[-2][-1:]
                gender, speaker_id, emotion = sentence_part[-3][:1], sentence_part[-3], label_dict[sentence_file]
                
                # we keep improv data only
                if recording_type == 'P' or recording_type == 'R': continue
                if emotion not in emo_map_dict: continue
                label, data = emo_map_dict[emotion], data_dict[sentence_file]
                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                save_data_dict(save_data, label, gender, speaker_id)
        elif data_set_str == 'crema-d':
            
            # speaker id for training and test
            train_speaker_id_arr, test_speaker_id_arr = [tmp_idx for tmp_idx in train_arr], [tmp_idx for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)

            # data root folder
            demo_df = pd.read_csv(str(Path(args.data_dir).joinpath('processedResults', 'VideoDemographics.csv')), index_col=0)
            rating_df = pd.read_csv(str(Path(args.data_dir).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
            sentence_file_list = list(Path(args.data_dir).joinpath('AudioWAV').glob('*.wav'))
            sentence_file_list.sort()
            
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_file = str(sentence_file).split('/')[-1].split('.wav')[0]
                sentence_part = sentence_file.split('_')
                speaker_id = int(sentence_part[0])
                emotion = rating_df.loc[sentence_file, 'MultiModalVote']
                
                if sentence_file not in data_dict: continue
                if emotion not in emo_map_dict: continue
                label, data = emo_map_dict[emotion], data_dict[sentence_file]
                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                gender = 'M' if demo_df.loc[int(sentence_part[0]), 'Sex'] == 'Male' else 'F'
                save_data_dict(save_data, label, gender, speaker_id)
        elif data_set_str == 'iemocap':
            # speaker id for training, validation, and test
            speaker_id_list = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M', 'Ses04F', 'Ses04M', 'Ses05F', 'Ses05M']
            train_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in train_arr]
            test_speaker_id_arr = [speaker_id_list[tmp_idx] for tmp_idx in test_arr]
            print('Train speaker:')
            print(train_speaker_id_arr)
            print('Test speaker:')
            print(test_speaker_id_arr)
        
            for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                ground_truth_path_list = list(Path(args.data_dir).joinpath(session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
                for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
                    with open(str(ground_truth_path)) as f:
                        file_content = f.read()
                        useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                        label_lines = re.findall(useful_regex, file_content)
                        for line in label_lines:
                            if 'Ses' in line:
                                sentence_file = line.split('\t')[-3]
                                gender = sentence_file.split('_')[-1][0]
                                speaker_id = sentence_file.split('_')[0][:-1] + gender
                                label, data = line.split('\t')[-2], data_dict[sentence_file]
                                save_data = np.array(data['data'])[0] if args.feature_type == 'emobase' else np.array(data['data'])[0, 0, :].flatten()
                                
                                if 'impro' not in line: continue
                                if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap' or label == 'exc':
                                    if label == 'exc': label = 'hap'
                                    save_data_dict(save_data, label, gender, speaker_id)
        save_non_iid(training_dict)
        save_test(test_dict)
        
        fold_idx += 1
        del training_dict, test_dict
