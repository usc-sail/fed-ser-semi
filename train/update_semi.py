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
import collections

class DatasetGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        data = self.dataset['data'][item]
        label = self.dataset['label'][item]
        return torch.tensor(data), torch.tensor(int(label))
    

class PeudoDatasetGenerator():
    def __init__(self, dataset, mode='train'):
        self.dataset = dataset
        self.mode = mode
        self.index = np.arange(len(dataset['data']))

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, item):
        data = torch.tensor(self.dataset['data'][item].copy())
        noise1 = torch.empty(self.dataset['data'][item].shape).normal_(mean=1, std=0.25)
        noise2 = torch.empty(self.dataset['data'][item].shape).normal_(mean=0, std=0.1)
        label = self.dataset['label'][item]
        data = data*noise1 + noise2
        return data, label
    
class WeakUnlabelDatasetGenerator():
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
        data = data*noise1 + noise2
        return data, label
    
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

    def update_weights_scaffold_fixmatch(self, model, history_model, c_global, c_local, labeled_data_dict, unlabeled_data_dict, u, logit_threshold, current_epoch, pseudo_data_dict, pseudo_dict, T):
        lr = float(self.args.learning_rate) / np.power(2, (int(current_epoch/100)))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
        global_model = copy.deepcopy(model)
        mse_criterion = nn.MSELoss().to(self.device)
        
        c_global_para, c_local_para = c_global.state_dict(), c_local.state_dict()
        cnt, step_outputs = 0, []
        
        # generate pseudo
        for iter in range(int(self.args.local_epochs)):
            # generate logits
            model.to(self.device)
            model.eval()
            
            # ----------------------------------------------
            # ----------------- multiview ------------------
            if iter == 0 and history_model and len(unlabeled_data_dict['data']) != 0:
                pseudo_label_arr = np.zeros([10, len(unlabeled_data_dict['data']), 4])
                for aug_idx in range(10):
                    dataset_unlabeled = WeakUnlabelDatasetGenerator(unlabeled_data_dict)
                    unlabeled_dataloader = DataLoader(dataset_unlabeled, batch_size=16, num_workers=0, shuffle=False)
                    logit_list, pseudo_real_label = [], []
                    
                    for batch_idx, batch_data in enumerate(unlabeled_dataloader):
                        x, y = batch_data
                        x = x.to(self.device)
                        
                        logits = model(x.float())
                        logits = F.softmax(logits/T, dim=1)
                        logit_arr = logits.detach().cpu().numpy()

                        for tmp_arr_idx in range(len(logits)):
                            logit_list.append(logit_arr[tmp_arr_idx])
                            pseudo_real_label.append(y[tmp_arr_idx])
                            
                    pseudo_label_arr[aug_idx, :, :] = np.array(logit_list)
                    pseudo_real_label = np.array(pseudo_real_label)

                # ----------------------------------------------
                # -------------------- ups ---------------------
                if current_epoch >= 300:
                    tau_p = logit_threshold  
                else:
                    tau_p = 0.5 + (logit_threshold - 0.5)*((current_epoch)/300)
                
                kappa_p = 0.005
                out_std, out_prob = np.std(pseudo_label_arr, axis=0), np.mean(pseudo_label_arr, axis=0)
                max_value, max_idx = np.max(out_prob, axis=1), np.argmax(out_prob, axis=1)
                max_std = []
                for idx, item in enumerate(max_idx): max_std.append(out_std[idx][item])
                max_std = np.array(max_std)
                
                selected_idx = (max_value >= tau_p) * (max_std < kappa_p)
                pseudo_target, pseudo_maxstd = max_idx[selected_idx], max_std[selected_idx]
                pseudo_idx = np.where(selected_idx == True)[0]
                
                # ----------------------------------------------
                class_dict = collections.Counter(labeled_data_dict['label'])
                
                # control how many samples we trust per class per pseudo-labeling
                sample_dict = {}
                for i in range(4): sample_dict[key] = 1
                    
                print(sample_dict, class_dict)
                if len(pseudo_target) > 0:
                    final_pseudo_list = []
                    for class_idx in range(4):
                        current_class_idx = np.where(pseudo_target==class_idx)
                        if len(np.where(pseudo_target==class_idx)[0]) > 0:
                            current_class_maxstd = pseudo_maxstd[current_class_idx]
                            sorted_maxstd_idx = np.argsort(current_class_maxstd)
                            if sample_dict[class_idx] == 0: continue
                            sorted_maxstd_idx = sorted_maxstd_idx[:sample_dict[class_idx]]
                            
                            for idx in sorted_maxstd_idx:
                                pseudo_data_dict['data'].append(unlabeled_data_dict['data'][pseudo_idx[idx]])
                                pseudo_data_dict['label'].append(class_idx)
                                
                                pseudo_dict['pseudo'].append(class_idx)
                                pseudo_dict['true'].append(pseudo_real_label[pseudo_idx[idx]])
                                print('add a new hard pseudo label')
                                final_pseudo_list.append(pseudo_idx[idx])
                                
                    if final_pseudo_list:
                        unlabeled_data_dict['data'] = np.delete(unlabeled_data_dict['data'], final_pseudo_list, axis=0)
                        unlabeled_data_dict['label'] = np.delete(unlabeled_data_dict['label'], final_pseudo_list)
            
            # add local weights to the training samples based on the class distribution
            class_dict = collections.Counter(labeled_data_dict['label'])
            minimum = min(class_dict, key=class_dict.get)
            cls_num_list = []
            for i in range(4):
                if i in class_dict: cls_num_list.append(class_dict[i])
                else: cls_num_list.append(class_dict[minimum])
            
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
            criterion = nn.NLLLoss(per_cls_weights).to(self.device)
            
            # training data loader
            dataset_train = WeakUnlabelDatasetGenerator(labeled_data_dict)
            train_dataloaders = DataLoader(dataset_train, batch_size=16, num_workers=0, shuffle=True)
            
            # unlabeled training data loader
            if len(pseudo_data_dict['label']) > 0:
                dataset_peudo = PeudoDatasetGenerator(pseudo_data_dict)
                peudo_dataloader = DataLoader(dataset_peudo, batch_size=16, num_workers=0, shuffle=False)
                pseudo_x_list, pseudo_y_list = [], []
                for batch_idx, batch_data in enumerate(peudo_dataloader):
                    pseudo_x, pseudo_y = batch_data
                    pseudo_x_list.append(pseudo_x)
                    pseudo_y_list.append(pseudo_y)
        
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
                logits = torch.log_softmax(logits, dim=1)
                loss = criterion(logits, y)
                
                if len(pseudo_data_dict['label']) > 0:
                    if batch_idx < len(peudo_dataloader):
                        logits1 = model(pseudo_x_list[batch_idx].to(self.device))
                        logits1 = torch.log_softmax(logits1, dim=1)
                        loss += criterion(logits1, pseudo_y_list[batch_idx].to(self.device))*len(logits1)/16
                        
                loss.backward()
                optimizer.step()
                cnt += 1
                
                # scaffold
                model_para = model.cpu().state_dict().copy()
                for key in model_para:
                    model_para[key] = model_para[key] - lr * (c_global_para[key] - c_local_para[key])
                model.load_state_dict(model_para)
                
                # obtain results
                predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                label_dist = label_dist + truth_list
                step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        
        # scaffold
        c_new_para = c_local.state_dict()
        c_delta_para = copy.deepcopy(c_local.state_dict())
        global_model_para = global_model.cpu().state_dict()
        local_model_para = model.cpu().state_dict()
        
        for key in local_model_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - local_model_para[key]) / (cnt * lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        c_local.load_state_dict(c_new_para)
        result_dict, _, _ = result_summary(step_outputs)
        data_list = []
        data_list.append(labeled_data_dict)
        data_list.append(unlabeled_data_dict)
        data_list.append(pseudo_data_dict)
        return model.state_dict(), c_delta_para, result_dict, data_list
    
    def inference(self, model):
        model.eval()
        step_outputs = []
        
        for batch_idx, batch_data in enumerate(self.dataloader):
            x, y = batch_data
            x, y = x.to(self.device), y.to(self.device)

            logits = model(x.float())
            logits = torch.log_softmax(logits, dim=1)
            loss = self.criterion(logits, y)
            
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
            truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
            step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict, y_true, y_pred = result_summary(step_outputs)
        return result_dict, y_true, y_pred

