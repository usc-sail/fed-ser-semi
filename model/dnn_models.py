#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
from torch.nn.modules import dropout
import itertools

class_dict = {'emotion': 4, 'affect': 3, 'gender': 2}

class dnn_classifier(nn.Module):
    def __init__(self, pred, input_spec, dropout):

        super(dnn_classifier, self).__init__()
        self.dropout_p = dropout
        self.num_classes = class_dict[pred]

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense1 = nn.Linear(input_spec, 256)
        self.dense2 = nn.Linear(256, 128)
        
        self.pred_layer = nn.Linear(128, self.num_classes)
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

    def forward(self, input_var):

        x = input_var.float()
        x = self.dense1(x)
        x = self.dense_relu1(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.dense_relu2(x)
        x = nn.Dropout(p=0.2)(x)

        preds = self.pred_layer(x)
        preds = torch.log_softmax(preds, dim=1)
        
        return preds
