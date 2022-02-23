import pandas as pd
from torch import nn
from torch._C import device
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import copy, pdb, time, warnings, torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from numpy import linalg as LA
import torch.nn.functional as F
warnings.filterwarnings('ignore')


class DatasetGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        data = self.dataset['data'][item]
        label = self.dataset['label'][item]
        return torch.tensor(data), torch.tensor(int(label))
    

class UnlabelDatasetGenerator():
    def __init__(self, dataset, mode='train'):
        self.dataset = dataset
        self.mode = mode
        self.index = np.arange(len(dataset['data']))

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        data = torch.tensor(self.dataset['data'][item].copy())
        noise1 = torch.empty(self.dataset['data'][item].shape).normal_(mean=1, std=0.1)
        noise2 = torch.empty(self.dataset['data'][item].shape).normal_(mean=0, std=0.1)
        label = self.dataset['label'][item]
        data = torch.dot(data, noise1) + noise2
        return data, label

class SelfDatasetGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        pos_data = torch.tensor(self.dataset['data'][item].copy())
        data = torch.tensor(self.dataset['data'][item].copy())
        noise1 = torch.empty(self.dataset['data'][item].shape).normal_(mean=1, std=0.1)
        noise2 = torch.empty(self.dataset['data'][item].shape).normal_(mean=0, std=0.1)
        label = self.dataset['label'][item]
        
        data = torch.dot(data, noise1) + noise2
        
        return pos_data, data
    
def pred_summary(y_true, y_pred):
    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    return result_dict

def average_weights(w, num_samples_list):
    """
    Returns the average of the weights.
    """
    total_num_samples = np.sum(num_samples_list)
    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        w_avg[key] = w[0][key]*(num_samples_list[0]/total_num_samples)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.div(w[i][key]*num_samples_list[i], total_num_samples)
    return w_avg

def result_summary(step_outputs):
    loss_list, y_true, y_pred = [], [], []
    for step in range(len(step_outputs)):
        for idx in range(len(step_outputs[step]['pred'])):
            y_true.append(step_outputs[step]['truth'][idx])
            y_pred.append(step_outputs[step]['pred'][idx])
        loss_list.append(step_outputs[step]['loss'])

    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    result_dict['loss'] = np.mean(loss_list)
    result_dict['num_samples'] = len(y_pred)
    result_dict['y_true'] = y_true
    return result_dict, y_true, y_pred

class local_trainer(object):
    def __init__(self, args, device, criterion, model_type, dataloader):
        self.args = args
        self.device = device
        self.dataloader = dataloader
        self.criterion = criterion
        self.model_type = model_type

    def update_weights_scaffold_unsupervised(self, model, history_model, c_global, c_local, labeled_data_dict, unlabeled_data_dict, u, logit_threshold):
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.learning_rate), weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
        global_model = copy.deepcopy(model)
        
        c_global_para, c_local_para = c_global.state_dict(), c_local.state_dict()
        cnt, step_outputs = 0, []
    
        # generate pseudo
        for iter in range(int(self.args.local_epochs)):
            # generate logits
            model.to(self.device)
            model.eval()
            if iter == 0 and history_model and len(unlabeled_data_dict['data']) > 0:
                pseudo_label_arr = np.zeros([5, len(unlabeled_data_dict['data']), 4])
                pseudo_hist_label_arr = np.zeros([5, len(unlabeled_data_dict['data']), 4])
                
                for aug_idx in range(5):
                    dataset_unlabeled = UnlabelDatasetGenerator(unlabeled_data_dict)
                    unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=16, num_workers=0, shuffle=False)
                    logit_list = []
                    history_logit_list = []
                    pseudo_real_label = []
                    
                    for batch_idx, batch_data in enumerate(unlabeled_dataloader):
                        # training the model
                        x, y = batch_data
                        x = x.to(self.device)
                        
                        logits = model(x.float())
                        logits = torch.exp(logits)
                        
                        history_logits = history_model(x.float())
                        history_logits = torch.exp(history_logits)
                        
                        logit_arr = logits.detach().cpu().numpy()
                        history_logit_arr = history_logits.detach().cpu().numpy()
                        
                        for tmp_arr_idx in range(len(logits)):
                            logit_list.append(logit_arr[tmp_arr_idx])
                            history_logit_list.append(history_logit_arr[tmp_arr_idx])
                            pseudo_real_label.append(y[tmp_arr_idx])
                            
                    pseudo_label_arr[aug_idx, :, :] = np.array(logit_list)
                    pseudo_hist_label_arr[aug_idx, :, :] = np.array(history_logit_list)
                    pseudo_real_label = np.array(pseudo_real_label)
                    
                # generate pseudo label
                avg_pseudo_logits, avg_hist_pseudo_logits = np.mean(pseudo_label_arr, axis=0), np.mean(pseudo_hist_label_arr, axis=0)
                max_pseudo_logits, max_hist_pseudo_logits = np.max(avg_pseudo_logits, axis=1), np.max(avg_hist_pseudo_logits, axis=1)
                valid_labels, valid_hist_labels = np.where(max_pseudo_logits > logit_threshold)[0], np.where(max_hist_pseudo_logits > logit_threshold)[0]
                # pdb.set_trace()
                if len(valid_labels) > 0 and len(valid_hist_labels) > 0:
                    final_pseudo_list = []
                    for idx in valid_labels:
                        if idx in valid_hist_labels:
                            labeled_data_dict['data'] = np.append(labeled_data_dict['data'], np.expand_dims(unlabeled_data_dict['data'][idx], axis=0), axis=0)
                            labeled_data_dict['label'] = np.append(labeled_data_dict['label'], unlabeled_data_dict['label'][idx])
                            final_pseudo_list.append(idx)
                    if final_pseudo_list:
                        unlabeled_data_dict['data'] = np.delete(unlabeled_data_dict['data'], final_pseudo_list, axis=0)
                        unlabeled_data_dict['label'] = np.delete(unlabeled_data_dict['label'], final_pseudo_list)
                    
            # training data loader
            dataset_train = DatasetGenerator(labeled_data_dict)
            train_dataloaders = DataLoader(dataset_train, batch_size=16, num_workers=0, shuffle=True)
            
            # unlabeled training data loader
            dataset_unlabeled = SelfDatasetGenerator(unlabeled_data_dict)
            unlabeled_batch_size = int(len(unlabeled_data_dict['data'])/len(train_dataloaders))
            if unlabeled_batch_size > 0:
                unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=unlabeled_batch_size, num_workers=0, shuffle=False)
                unlabeled_x_list, unlabeled_x_pos_list = [], []
                for batch_idx, batch_data in enumerate(unlabeled_dataloader):
                    unlabeled_x, unlabeled_x_pos = batch_data
                    unlabeled_x_list.append(unlabeled_x)
                    unlabeled_x_pos_list.append(unlabeled_x_pos)
            
            model.train()
            # first training
            label_dist = []
            for batch_idx, batch_data in enumerate(train_dataloaders):
                
                # training the model
                model.to(self.device)
                model.zero_grad()
                optimizer.zero_grad()
                
                x, y = batch_data
                x, y = x.to(self.device), y.to(self.device)
                
                logits = model(x.float())
                loss = self.criterion(logits, y)
                
                if unlabeled_batch_size > 0:
                    logits1 = model(unlabeled_x_list[batch_idx].to(self.device))
                    logits2 = model(unlabeled_x_pos_list[batch_idx].to(self.device))
                    logits1, logits2 = torch.exp(logits1), torch.exp(logits2)
                    loss += u*F.kl_div(logits1, logits2)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                
                # scaffold
                model_para = model.cpu().state_dict().copy()
                for key in model_para:
                    model_para[key] = model_para[key] - float(self.args.learning_rate) * (c_global_para[key] - c_local_para[key])
                model.load_state_dict(model_para)
                
                # obtain results
                predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                label_dist = label_dist + truth_list
                step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})

        c_new_para = c_local.state_dict()
        c_delta_para = copy.deepcopy(c_local.state_dict())
        global_model_para = global_model.cpu().state_dict()
        local_model_para = model.cpu().state_dict()
        
        for key in local_model_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - local_model_para[key]) / (cnt * float(self.args.learning_rate))
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        c_local.load_state_dict(c_new_para)
        result_dict, _, _ = result_summary(step_outputs)
        
        data_list = []
        data_list.append(labeled_data_dict)
        data_list.append(unlabeled_data_dict)
        return model.state_dict(), c_delta_para, result_dict, data_list
    
    
    def inference(self, model):
        model.eval()
        step_outputs = []
        
        for batch_idx, batch_data in enumerate(self.dataloader):
            x, y = batch_data
            x, y = x.to(self.device), y.to(self.device)

            logits = model(x.float())
            loss = self.criterion(logits, y)
            
            logits = torch.exp(logits)
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
            truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
            step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict, y_true, y_pred = result_summary(step_outputs)
        return result_dict, y_true, y_pred
    
    def get_emb(self, model):
        model.eval()
        z_list, y_list = [], []
        for batch_idx, batch_data in enumerate(self.dataloader):
            x, y = batch_data
            logits = model(x.to(self.device))
            # obtrain results
            y = y.detach().cpu().numpy()
            for pred_idx in range(len(y)):
                y_list.append(y[pred_idx])
            
        return z_list, y_list
