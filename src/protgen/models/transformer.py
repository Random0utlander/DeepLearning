import torch
from torch import nn, softmax
import math
from torch.utils.data import Dataset
import transformers

class Transformer(nn.Module):
    def __init__(self, vocab_size=23, dim_embedding=10, num_hidden=50, max_seq_len=129):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_seq_len, dim_embedding)
        self.Q = nn.Linear(dim_embedding, dim_embedding) #d_q tem de ser d_k
        self.K = nn.Linear(dim_embedding, dim_embedding)
        self.V = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, x):
        x = self.embeddings(x)
        # Add positional embeddings
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.positional_embeddings(pos)
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)

        # Return the output of the attention
        attention_map = Q @ K.transpose(-2, -1) / math.sqrt(float(K.shape[-1]))

        #proibe que a atenção olhe para o futuro
        casual_mask = torch.triu(torch.ones(attention_map.size(-2), attention_map.size(-1), device=attention_map.device), diagonal=1).bool()

        #substitui os valores da atenção que olham para o futuro por -inf, assim a softmax vai dar 0 a esses valores
        attention_map = attention_map.masked_fill(casual_mask, float('-inf'))

        attention_weights = softmax(attention_map, dim=-1)

        output=attention_weights @ V

        return output

        
# não usei, mas acho que é preciso usar
def tokenize (protein_dataset):
    """Tokenizes a protein sequence into a tensor of integers"""
    tokens=protein_dataset.tokens
    return tokens





