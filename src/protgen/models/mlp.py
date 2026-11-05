import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, vocab_size=23, dim_embedding=10, num_hidden=50, window_size=3):
        # Vocab size is 23 for the 20 amino acids + 3 special tokens (start <BOS>, end <EOS>, padding <PAD>)
        # dim_embedding is the dimension of the embedding vector for each amino acid
        # num_hidden is the number of hidden units in the first layer
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.fc1 = nn.Linear(window_size*dim_embedding, num_hidden)
        self.fc2 = nn.Linear(num_hidden, vocab_size)

    def forward(self, x):
       
        # Id to embedding (batch, 3) -> (batch, 3, 10)
        x = self.embedding(x)
       
        # flatten the embedding (batch, 3, 10) -> (batch, 30)
        x = x.flatten(start_dim=1)

        # first layer (batch, 30) -> (batch, 50)
        x = self.fc1(x)
       
        #Activation function
        x = F.relu(x)
        
        # second layer (batch, 50) -> (batch, 23)
        x = self.fc2(x)
        return x


