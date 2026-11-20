import torch
from torch import nn, softmax
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size=23, dim_embedding=64, num_hidden=256, max_seq_len=129,dropout=0.1):
        # a causal decoder-only multi-head attention Transformer model, I was doing single head attention but the objective is multi.head attention 
        # see if it necessary to add dropout in case of overfitting, but for now I will not add it
        super().__init__()


        self.embeddings = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_seq_len, dim_embedding)
        self.dim_head = 16
        self.num_heads = dim_embedding // self.dim_head
        
        assert dim_embedding % self.dim_head == 0, (
           "dim_embedding must be divisible by dim_head"
            )

        #Dropouts
        self.emb_dropout   = nn.Dropout(dropout)
        self.attn_dropout  = nn.Dropout(dropout)   # on attention weights
        self.resid_dropout = nn.Dropout(dropout)   # on attention output
        self.ffn_dropout   = nn.Dropout(dropout)   # inside + after FFN
        
        #Atention
        self.Q = nn.Linear(dim_embedding, self.dim_head * self.num_heads) #d_q need to be = d_k
        self.K = nn.Linear(dim_embedding, self.dim_head * self.num_heads)
        self.V = nn.Linear(dim_embedding, self.dim_head * self.num_heads)
        
        self.norm_f=nn.LayerNorm(dim_embedding)
        self.output_projection= nn.Linear(dim_embedding, dim_embedding)
        self.ffn1= nn.Linear(dim_embedding, num_hidden)
        self.ffn2= nn.Linear(num_hidden, dim_embedding)
        self.norm_ffn=nn.LayerNorm(dim_embedding)
        self.language_model_head= nn.Linear(dim_embedding, vocab_size)


    def forward(self, x):
        x = self.embeddings(x)
        # Add positional embeddings
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x =self.emb_dropout(x + self.positional_embeddings(pos))

        #Q, K, V matrices
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)

        Q_reshaped=Q.view(Q.size(0), Q.size(1), self.num_heads, self.dim_head).transpose(1, 2) # (batch_size, num_heads, seq_len, dim_head)
        k_reshaped=K.view(K.size(0), K.size(1), self.num_heads, self.dim_head).transpose(1, 2) # (batch_size, num_heads, seq_len, dim_head)
        v_reshaped=V.view(V.size(0), V.size(1), self.num_heads, self.dim_head).transpose(1, 2) # (batch_size, num_heads, seq_len, dim_head)
        
        # Return the output of the attention
        attention_map = Q_reshaped @ k_reshaped.transpose(-2, -1) / math.sqrt(float(k_reshaped.shape[-1]))

        #proihibits the attention from looking into the future, so that the model can only attend to previous tokens in the sequence
        casual_mask = torch.triu(torch.ones(attention_map.size(-2), attention_map.size(-1), device=attention_map.device), diagonal=1).bool()

        #substits the values of the attention that look to the future by -inf, so the softmax will give 0 to those values
        attention_map = attention_map.masked_fill(casual_mask, float('-inf'))

        attention_weights = self.attn_dropout(softmax(attention_map, dim=-1))

        output=attention_weights @ v_reshaped

        # "Concat" using a view (batch_size, seq_len, dim_head * num_heads)
        output_reshaped_back= output.transpose(1, 2).contiguous().view(output.size(0), -1, self.dim_head * self.num_heads) 
        
        # Project back to the original embedding dimension
        output_projection = self.resid_dropout(self.output_projection(output_reshaped_back))

        residual_connection = x + output_projection

        output_norm_f = self.norm_f(residual_connection)

        output_ffn = self.feed_forward_network(output_norm_f)

        output_f = self.language_model_head(output_ffn)

        return output_f #(batch of proteins, max_seq_len, vocab_size)

    def feed_forward_network(self, x):
        # Feed-forward network dim=64 -> 256 -> 64; similiar to the MLP model, but with residual connection and layer normalization
        h = nn.functional.relu(self.ffn1(x))

        #Dropout after first layer
        h = self.ffn_dropout(h)

        h = self.ffn2(h)
        
        #Dropout after second layer
        h = self.ffn_dropout(h)
                                # Residual connection
        ffn_output_f = self.norm_ffn(h + x) # Second layer normalization

        return ffn_output_f
