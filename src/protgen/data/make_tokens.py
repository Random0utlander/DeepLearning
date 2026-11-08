import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .make_data import IGNORE_INDEX


class TokenDataset(Dataset):
    """
    Dataset for creating the tokens for the MLP input.

    Samples formating: 
        x: The input (A context window of 3 (default) tokens)
        y: The Target aminoacid (the next token for the mlp to predict)
    """

    def __init__(self, protein_dataset, window_size=3):

        # Get the encoded sequences and corresponding targets
        tokens = protein_dataset.tokens
        targets = protein_dataset.tokens_shifted

        # Pad the left side of each sequence with (Window size - 1)  PAD tokens
        tokens_padded = F.pad(tokens, (window_size-1, 0), "constant", 0)

        # Create the sliding window inputs for the MLP
        windows = tokens_padded.unfold(
            dimension=1,
            size=window_size,
            step=1
        )

        # Boolean mask for valid targets (exclude IGNORE_INDEX)
        mask = (targets != IGNORE_INDEX)
        self.targets = targets[mask]
        self.inputs = windows[mask]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]