import torch 
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch import optim
from protgen.models.transformer import Transformer
from protgen.data import ProteinDataset
from protgen.data.make_tokens import TokenDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
import matplotlib.pyplot as plt
import os

# Load the best MLP model
best_transformer = torch.load("../src/protgen/models/transformer_best.pt", weights_only=True)

# i forgot to add vocab size to the saved model, so we need to calculate it here
VOCAB_SIZE = len(special_tokens) + len(AA) 

#Create the MLP model 
transformer = Transformer(
        vocab_size=VOCAB_SIZE,
        dim_embedding=best_transformer["dim_embedding"],
        num_hidden=best_transformer["hidden_units"],
        dim_head=best_transformer["head_dim"],
        dropout=best_transformer["dropout"]
    )

# Load the weights of the best model into the MLP instance
transformer.load_state_dict(best_transformer["weights"])

# Set the model to evaluation mode
transformer.eval()