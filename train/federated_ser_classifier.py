from moviepy.tools import verbose_print
import torch
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

import numpy as np
from pathlib import Path
import pandas as pd
import copy, time, pickle, shutil, sys, os, pdb
from copy import deepcopy

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))

from dnn_models import dnn_classifier
from update import average_weights, average_gradients, local_trainer, pred_summary
                       
# define feature len mapping
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

def save_result(save_index, acc, uar, best_epoch, dataset):
    row_df = pd.DataFrame(index=[save_index])
    row_df['acc'], row_df['uar'], row_df['epoch'], row_df['dataset']  = acc, uar, best_epoch, dataset
    return row_df

class DatasetGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        data = self.dataset['data'][item]
        label = self.dataset['label'][item]
        return torch.tensor(data), torch.tensor(int(label))

def read_data_dict_by_client(fold_idx):
    return_labeled_train_dict, return_unlabeled_train_dict, return_val_dict, return_test_dict = {}, {}, {}, {}
    # prepare the data for the training
    file_idx = 0
    for file_name in os.listdir(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'train')):
        speaker_id = file_name.split('.pkl')[0]
        with open(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'train', file_name), 'rb') as f:
            data_dict = pickle.load(f)
        
        # how many data we have for training
        perm_array = np.random.RandomState(seed=file_idx).permutation(len(data_dict['label']))
        file_idx += 1
        x, y = data_dict['data'], data_dict['label']
        return_labeled_train_dict[speaker_id] = {}
        return_labeled_train_dict[speaker_id]['data'] = x[perm_array[:int(float(args.client_label_rate)*len(y))]]
        return_labeled_train_dict[speaker_id]['label'] = y[perm_array[:int(float(args.client_label_rate)*len(y))]]
        
        return_unlabeled_train_dict[speaker_id] = {}
        return_unlabeled_train_dict[speaker_id]['data'] = x[perm_array[int(float(args.client_label_rate)*len(y)):]]
        return_unlabeled_train_dict[speaker_id]['label'] = y[perm_array[int(float(args.client_label_rate)*len(y)):]]
        
    for file_name in os.listdir(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'validation')):
        speaker_id = file_name.split('.pkl')[0]
        with open(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'validation', file_name), 'rb') as f:
            data_dict = pickle.load(f)
        return_val_dict[speaker_id] = {}
        return_val_dict[speaker_id] = data_dict.copy()
        
    for file_name in os.listdir(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'test')):
        speaker_id = file_name.split('.pkl')[0]
        with open(preprocess_path.joinpath(args.dataset, 'fold'+str(int(fold_idx+1)), 'test', file_name), 'rb') as f:
            data_dict = pickle.load(f)
        return_test_dict[speaker_id] = {}
        return_test_dict[speaker_id] = data_dict.copy()
    return return_labeled_train_dict, return_unlabeled_train_dict, return_val_dict, return_test_dict

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--learning_rate', default=0.05)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=500)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--client_label_rate', default=0.4)
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')
    args = parser.parse_args()

    preprocess_path = Path(args.save_dir).joinpath('fed-ser-semi', args.feature_type, args.pred)
    
    # set seeds
    save_result_df = pd.DataFrame()

    # find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # We perform 5 fold experiments
    for fold_idx in range(5):
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])
        
        model_setting_str = 'local_epoch_'+str(args.local_epochs)
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        model_setting_str += '_lr_' + str(args.learning_rate)[2:]
        model_setting_str += '_client_label_rate_' + str(args.client_label_rate).replace('.', '')
        
        # Read the data per speaker
        train_labeled_speaker_dict, train_unlabeled_speaker_dict, val_speaker_dict, test_speaker_dict = read_data_dict_by_client(fold_idx)
        num_of_speakers, speaker_list = len(train_labeled_speaker_dict), list(set(train_labeled_speaker_dict.keys()))
        
        # Define the global model
        global_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout))
        global_model = global_model.to(device)
        global_weights = global_model.state_dict()
        
        # Define scaffold model
        c_model = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout))
        c_local_dict = {}
        for idx in range(num_of_speakers): 
            c_local_dict[speaker_list[idx]] = dnn_classifier(pred='emotion', input_spec=feature_len_dict[args.feature_type], dropout=float(args.dropout))
        
        # copy weights
        criterion = nn.NLLLoss().to(device)
        
        # log saving path
        if args.model_type == 'fed_avg':
            model_result_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'supervised', args.dataset, args.feature_type, model_setting_str)
        else:
            model_result_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', 'scaffold_supervised', args.dataset, args.feature_type, model_setting_str)
        
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        Path.mkdir(model_result_path.joinpath(save_row_str), parents=True, exist_ok=True)

        # Training steps
        result_dict, best_score = {}, 0
        seed_everything(8, workers=True)
        torch.manual_seed(8)
        np.random.seed(8)
        random_seed = 8
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        for epoch in range(int(args.num_epochs)):
            # we choose 20% of clients in training
            np.random.seed(epoch)
            idxs_speakers = np.random.choice(range(num_of_speakers), int(0.1 * num_of_speakers), replace=False)
            
            # define list varibles that saves the weights, loss, num_sample, etc.
            local_updates, local_c_deltas, local_losses, local_num_sampels = [], [], [], []
            
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            for idx in idxs_speakers:
                speaker_id = speaker_list[idx]
                
                # 1.1 Local training
                dataset_train = DatasetGenerator(train_labeled_speaker_dict[speaker_id])
                train_dataloaders = DataLoader(dataset_train, batch_size=16, num_workers=0, shuffle=True)
                trainer = local_trainer(args, device, criterion, args.model_type, train_dataloaders)
           
                # read shared updates: parameters in fed_avg and gradients for fed_sgd
                if args.model_type == 'scaffold':
                    local_update, local_c_delta, train_result = trainer.update_weights_scaffold(model=copy.deepcopy(global_model), c_global=copy.deepcopy(c_model), c_local=c_local_dict[speaker_id])
                    local_c_deltas.append(local_c_delta)
                else:
                    local_update, train_result = trainer.update_weights(model=copy.deepcopy(global_model))
                local_updates.append(copy.deepcopy(local_update))
                
                # read params to save
                local_losses.append(train_result['loss'])
                local_num_sampels.append(train_result['num_samples'])

                print('speaker id %s sample size %d' % (speaker_id, train_result['num_samples']))
                del trainer
            
            # 2. global model updates
            total_num_samples = np.sum(local_num_sampels)
            # 2.1 average global weights
            global_weights = average_weights(local_updates, local_num_sampels)
            # 2.2 load new global weights
            global_model.load_state_dict(global_weights)
            
            # 2.3 scaffold
            if args.model_type == 'scaffold':
                c_delta_weights = average_weights(local_c_deltas, np.ones(len(local_c_deltas)))
                c_global_para = c_model.state_dict()
                for key in c_global_para: c_global_para[key] += c_delta_weights[key]
                c_model.load_state_dict(c_global_para)

            # 3. Calculate avg validation accuracy/uar over all selected users at every epoch
            validation_acc, validation_uar, validation_loss, local_num_sampels = [], [], [], []
            # 3.1 Iterate each client at the current global round, calculate the performance
            for idx in range(num_of_speakers):
                speaker_id = speaker_list[idx]
                dataset_validation = DatasetGenerator(val_speaker_dict[speaker_id])
                val_dataloaders = DataLoader(dataset_validation, batch_size=16, num_workers=0, shuffle=False)
                
                trainer = local_trainer(args, device, criterion, args.model_type, val_dataloaders)
                local_val_result, _, _ = trainer.inference(copy.deepcopy(global_model))
                
                # save validation accuracy, uar, and loss
                local_num_sampels.append(local_val_result['num_samples'])
                validation_acc.append(local_val_result['acc'])
                validation_uar.append(local_val_result['uar'])
                validation_loss.append(local_val_result['loss'])
                del val_dataloaders, trainer
            
            # 3.2 Re-Calculate weigted performance scores
            validate_result = {}
            weighted_acc, weighted_rec = 0, 0
            total_num_samples = np.sum(local_num_sampels)
            for acc_idx in range(len(validation_acc)):
                weighted_acc += validation_acc[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
                weighted_rec += validation_uar[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
            validate_result['acc'], validate_result['uar'] = weighted_acc, weighted_rec
            validate_result['loss'] = np.mean(validation_loss)

            print('------------------------------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------------------------------')
            print('| Global Round validation : {} | acc: {:.2f}% | uar: {:.2f}% | Loss: {:.6f}'.format(
                        epoch, weighted_acc*100, weighted_rec*100, train_result['loss']))
            print('------------------------------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------------------------------')
            
            # 4. Perform the test on holdout set
            test_preds, test_truths = [], []
            for speaker_id in test_speaker_dict:
                # test loader
                dataset_test = DatasetGenerator(test_speaker_dict[speaker_id])
                test_dataloaders = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False)

                trainer = local_trainer(args, device, criterion, args.model_type, test_dataloaders)
                _, truth, preds = trainer.inference(copy.deepcopy(global_model))
                
                test_preds, test_truths = test_preds + list(preds), test_truths + list(truth)
                del test_dataloaders
            test_result = pred_summary(test_truths, test_preds)
            
            # 5. Save the results for later
            result_dict[epoch] = {}
            result_dict[epoch]['train'] = {}
            result_dict[epoch]['train']['loss'] = sum(local_losses) / len(local_losses)
            result_dict[epoch]['validate'] = validate_result
            result_dict[epoch]['test'] = test_result
            
            if epoch == 0: best_epoch, best_val_dict, best_test_dict = 0, validate_result, test_result
            if validate_result['uar'] > best_val_dict['uar'] and epoch > 0:
                # Save best model and training history
                best_epoch, best_val_dict, best_test_dict = epoch, validate_result, test_result
                torch.save(deepcopy(global_model.state_dict()), str(model_result_path.joinpath(save_row_str, 'model.pt')))
            
            if epoch > 0:
                # log results
                print('------------------------------------------------------------------------------------------------------')
                print('------------------------------------------------------------------------------------------------------')
            
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, best_test_dict['acc']*100, best_val_dict['acc']*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, best_test_dict['uar']*100, best_val_dict['uar']*100))
                print(best_test_dict['conf'])
        
        # Performance save code
        row_df = save_result(save_row_str, best_test_dict['acc'], best_test_dict['uar'], best_epoch, args.dataset)
        save_result_df = pd.concat([save_result_df, row_df])
        save_result_df.to_csv(str(model_result_path.joinpath('result.csv')))
        
        f = open(str(model_result_path.joinpath(save_row_str, 'results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

    # Calculate the average of the 5-fold experiments
    tmp_df = save_result_df.loc[save_result_df['dataset'] == args.dataset]
    row_df = save_result('average', np.mean(tmp_df['acc']), np.mean(tmp_df['uar']), best_epoch, args.dataset)
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(model_result_path.joinpath('result.csv')))
