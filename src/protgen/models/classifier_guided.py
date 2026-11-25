import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import softmax
import math


class Classifier(nn.Module):

    def __init__(self, backbone, num_features, num_hidden=128, num_output=3):
        super(Classifier, self).__init__()  
        
        self.backbone = backbone #Transformer as base

        # input layer
        self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))
        # hidden layer
        self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_output, num_hidden)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_output), 0))
        # define activation function in constructor
        self.activation = torch.nn.ELU()

        #freeze parameters of Transformer
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._captured = {}
        self.backbone.language_model_head.register_forward_pre_hook(
            lambda model, input: self._captured.__setitem__("hidden",input[0])
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    
    def forward(self, x, lengths=None):

        #pass input through Transformer without optimizing the parameters
        with torch.no_grad():    
            self.backbone(x)  

        #save the intermediate vectors from the transformer pass
        #to pass through the classifier later 
        hidden = self._captured["hidden"] #batch_size x seq_len x embed_dim
        seq_len = hidden.size(1)
        positions = torch.arange(seq_len, device=hidden.device) 

        if lengths is None:
            lengths = (x != 0).sum(dim=1)
        lengths = lengths.clamp(min=1)


        mask = (positions[None, :] < lengths[:, None]).unsqueeze(-1).float()
        #True or False, size (batch_size x seq_len x 1)

        pooled = (hidden * mask).sum(dim=1) / lengths[:, None].float()

        # so batch_size x 129 x 3
        #x = hidden.flatten(start_dim=1)
        x = F.linear(pooled, self.W_1, self.b_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2, self.b_2)        
        return x
