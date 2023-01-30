import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from transformers import GPT2LMHeadModel, GPT2Model
from transformers import GPT2Tokenizer
from transformers import BertForSequenceClassification, BertModel
import copy
from collections import OrderedDict
from torch.nn import init
from transformers import DistilBertForSequenceClassification, DistilBertModel


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(CNN,self).__init__()
        self.n_outputs = n_outputs
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs)
    def forward(self,x):
        feature=self.fc(self.conv(x).view(x.shape[0], -1))
        return feature

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, n_outputs):
        super(ResNet, self).__init__()
        self.n_outputs = n_outputs
        # self.network = torchvision.models.resnet18(pretrained=True)
        self.network = torchvision.models.resnet50(pretrained=True)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        self.dropout = nn.Dropout(0)
        self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class CatClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_shards):
        super(CatClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_shards = num_shards
        self.network = torch.nn.Linear(in_features, out_features * num_shards)

    def aggregate(self, coeff):
        weight = (coeff @ self.network.state_dict()['weight'].reshape(self.num_shards, self.out_features * self.in_features)).reshape(self.out_features, self.in_features)
        bias = (coeff @ self.network.state_dict()['bias'].reshape(self.num_shards, self.out_features)).reshape(self.out_features)
        # for i, w in enumerate(coeff.squeeze()):
        #     if i == 0:
        #         weight_2 = w * self[i]['weight']
        #         bias_2 = w * self[i]['bias']
        #     else:
        #         weight_2 += w * self[i]['weight']
        #         bias_2 += w * self[i]['bias']
        # print(weight - weight_2)
        # print(bias - bias_2)
        return weight, bias

    def __getitem__(self, n):
        weights = self.network.state_dict()['weight'][self.out_features*n:self.out_features*(n+1),:]
        bias = self.network.state_dict()['bias'][self.out_features*n:self.out_features*(n+1)]
        return OrderedDict({'weight': weights,"bias": bias})
    
    def concat(self, local_classifier_list):
        concat_weight = []
        concat_bias = []
        for classifier in local_classifier_list:
            concat_weight.append(classifier['weight'])
            concat_bias.append(classifier['bias'])
        
        concat_weight = torch.cat(concat_weight, dim=0)
        concat_bias = torch.cat(concat_bias, dim=0)
        concat_param = OrderedDict({'weight': concat_weight,"bias": concat_bias})
        self.network.load_state_dict(concat_param)
    
    def forward(self, x):
        return self.network(x)
        

class InvariantClassifier(nn.Module):
    def __init__(self, in_envs, bias=False):
        super(InvariantClassifier, self).__init__()
        self.in_envs = in_envs
        self.network = torch.nn.Linear(self.in_envs, 1, bias=bias)
        weight = 1/self.in_envs *  torch.ones((1, self.in_envs))
        if bias:
            bias_param = torch.zeros((1))
            self.network.load_state_dict(OrderedDict({"weight": weight, "bias": bias_param}))
        else:
            self.network.load_state_dict(OrderedDict({"weight": weight}))
    
    def forward(self, x):
        z = torch.stack(torch.split(x, int(x.shape[1]/self.in_envs), dim=1), dim=2)
        z = self.network(z).squeeze(dim=2)
        return z