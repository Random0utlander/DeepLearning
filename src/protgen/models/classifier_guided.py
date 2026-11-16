import torch
from torch.mtia import init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import softmax
import math


class Classifier(nn.Module):

    def __init__(self, backbone, num_features, num_hidden=128, num_output=1):
        super(Classifier, self).__init__()  
        
        self.backbone = backbone
        self.head = nn.Sequential(
            # input layer
            Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features))),
            Parameter(init.constant_(torch.Tensor(num_hidden), 0)),
            # hidden layer
            Parameter(init.xavier_normal_(torch.Tensor(num_output, num_hidden))),
            Parameter(init.constant_(torch.Tensor(num_output), 0)),
            # define activation function in constructor
            torch.nn.ELU()
        )

        for p in self.backbone.parameter():
            p.requires_grad = False

        self._captured = {}
        self.backbone.feed_forward_network.register_foward_hook(
            lambda model, input, output: self._captured.__setitem__("hidden",output)
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    
    def forward(self, x):
        with torch.no_grad():    
            self.backbone(x)
        hidden = self._captured["hidden"]

        #pool
        
        return x
