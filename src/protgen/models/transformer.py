import torch
from torch import nn, softmax
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size=23, dim_embedding=12, num_hidden=50, max_seq_len=129):
        # a causal decoder-only multi-head attention Transformer model, I was doing single head attention but the objective is multi.head attention 
        # see if it necessary to add dropout in case of overfitting, but for now I will not add it
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim_embedding, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_seq_len, dim_embedding)
        self.num_heads = 3
        self.dim_head = 4
        self.Q = nn.Linear(dim_embedding, self.dim_head * self.num_heads) #d_q tem de ser = d_k
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
        x = x + self.positional_embeddings(pos)

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

        attention_weights = softmax(attention_map, dim=-1)

        output=attention_weights @ v_reshaped

        # "Concat" using a view (batch_size, seq_len, dim_head * num_heads)
        output_reshaped_back= output.transpose(1, 2).contiguous().view(output.size(0), -1, self.dim_head * self.num_heads) 
        
        # Project back to the original embedding dimension
        output_projection = self.output_projection(output_reshaped_back)

        residual_connection = x + output_projection

        output_norm_f = self.norm_f(residual_connection)

        output_ffn = self.feed_forward_network(output_norm_f)

        output_f = self.language_model_head(output_ffn)

        return output_f

    def feed_forward_network(self, x):
        # Feed-forward network dim=12 -> 50 -> 12; similiar to the MLP model, but with residual connection and layer normalization
        ffn_output = self.ffn1(x)

        ffn_output = nn.ReLU()(ffn_output)

        ffn_output = self.ffn2(ffn_output)

        ffn_output= ffn_output + x # Residual connection

        ffn_output_f = self.norm_ffn(ffn_output) # Second layer normalization

        return ffn_output_f


#Chatgpt code, just for verification of the dimensions, I will take it out later
model = Transformer()

# Batch fictício: 32 proteínas com 129 tokens cada
x = torch.randint(0, 23, (32, 129))

# Verificar embeddings
embeddings = model.embeddings(x)
print("Embeddings:", embeddings.shape)

# Verificar output final do Transformer
output = model(x)
print("Transformer output:", output.shape)