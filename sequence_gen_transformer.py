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


BATCH_SIZE = 500
MAX_LENGTH = 129

# Load the best Transformer model
best_transformer = torch.load(
    "./src/protgen/models/transformer_best.pt", weights_only=True
)

# i forgot to add vocab size to the saved model, so we need to calculate it here
VOCAB_SIZE = len(special_tokens) + len(AA)

# Create the Transformer model
transformer = Transformer(
    vocab_size=VOCAB_SIZE,
    dim_embedding=best_transformer["dim_embedding"],
    num_hidden=best_transformer["hidden_units"],
    dim_head=best_transformer["head_dim"],
    dropout=best_transformer["dropout"],
    max_seq_len=MAX_LENGTH,
)

# Load the weights of the best model into the Transformer instance
transformer.load_state_dict(best_transformer["weights"])

# Set the model to evaluation mode
transformer.eval()

# Create a BATCH_SIZE*MAX_LENGTH input matrix and first colum to o <BOS> (1)
input = torch.zeros(BATCH_SIZE, MAX_LENGTH, dtype=int)
input[:, 0] = 1

with torch.no_grad():
    for i in range(1, MAX_LENGTH):  # index runs from 1 to 128 inclusive
        logits = transformer(input)  # Re-feed next column
        probabilities = torch.softmax(
            logits[:, i - 1, :], dim=-1
        )  # Logits -> Probabilities
        input[:, i] = torch.multinomial(probabilities, num_samples=1).squeeze(
            1
        )  # Sample Column along all sequences and update input


# Mask all the positions in front of the first <EOS> (2) 
is_two = (input == 2).long()
after_two = (is_two.cumsum(dim=1) - is_two )> 0
output = input.masked_fill(after_two, 0)


def decode(tokens):
    """Convert a tensor of token ids to a string of letters

    Only the 20 residues have a character: <BOS>, <EOS>, <PAD> and the IGNORE_INDEX of a
    shifted target are dropped, so decoding either x or x_shifted gives the protein back.
    """
    total_tokens = list(special_tokens) + list(AA)
    vocab = dict(zip(total_tokens, range(len(total_tokens))))
    idx_to_vocab = {index: token for token, index in vocab.items()}
    residues = set(AA)
    return "".join(
        idx_to_vocab[int(token)]
        for token in tokens
        if idx_to_vocab.get(int(token)) in residues
    )


sequences = [decode(seq) for seq in output]
pd.DataFrame(sequences, columns=["sequence"]).to_csv(
    "./results/sequences/transformer_generated_sequences.csv", index=False
)
