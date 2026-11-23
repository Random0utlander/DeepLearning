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

BATCH_SIZE = 500
MAX_LENGTH = 129

# Load the best MLP model
best_transformer = torch.load("./src/protgen/models/transformer_best.pt", weights_only=True)

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

# Load the weights of the best model into the Transformer instance
transformer.load_state_dict(best_transformer["weights"])

# Set the model to evaluation mode
transformer.eval()

input = torch.zeros(BATCH_SIZE,MAX_LENGTH, dtype=int)
input[:,0]=1


logits = transformer(input)

probabilities = torch.softmax(logits, dim=-1)

tokens = torch.zeros_like(input)

for i in range(MAX_LENGTH):
    sampled_tokens = torch.multinomial(probabilities[:,i,:], num_samples=1)
    tokens[:, i] = sampled_tokens[:,0]



def decode(tokens):
    """Convert a tensor of token ids to a string of letters

    Only the 20 residues have a character: <BOS>, <EOS>, <PAD> and the IGNORE_INDEX of a
    shifted target are dropped, so decoding either x or x_shifted gives the protein back.
    """
    total_tokens = list(special_tokens) + list(AA)
    vocab = dict(zip(total_tokens, range(len(total_tokens))))
    idx_to_vocab = {index: token for token, index in vocab.items()}
    residues = set(AA)
    return "".join(idx_to_vocab[int(token)] for token in tokens
                    if idx_to_vocab.get(int(token)) in residues)

sequences = [decode(seq) for seq in tokens]
print(f"Decoded sequences: {sequences}")
pd.DataFrame(sequences, columns=["sequence"]).to_csv("./results/sequences/transformer_generated_sequences.csv", index=False)