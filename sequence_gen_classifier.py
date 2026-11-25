import torch
from protgen.models.transformer import Transformer
from protgen.models.classifier_guided import Classifier
from protgen.data import ProteinDataset
from protgen.data.make_tokens import TokenDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
import matplotlib.pyplot as plt
import pandas as pd

val_data = ProteinDataset(path_to_data="./data/val.csv")
val_loader = DataLoader(val_data, batch_size=32)
BATCH_SIZE = 30
MAX_LENGTH = 129
Lambda = 0
FAM = 0
print(f"Generated guided sequence to family to: {FAMILIES[FAM]}")

# Load the best Transformer model
best_transformer = torch.load(
    "./src/protgen/models/transformer_best.pt", weights_only=True
)

# Load the best Classifier model
best_classifier = torch.load(
    "./src/protgen/models/classifier.pt", weights_only=True
)


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

#Create the classifier 
classifier = Classifier(
    backbone=transformer,
    num_features=best_transformer["dim_embedding"],
    num_hidden=128,
)

classifier.load_state_dict(best_classifier)

#set the model to evaluation mode
transformer.eval()

# Create a BATCH_SIZE*MAX_LENGTH input matrix and first colum to o <BOS> (1)
input = torch.zeros(BATCH_SIZE, MAX_LENGTH, dtype=int)
input[:, 0] = 1

transformer.eval()
x = next(iter(val_loader))[0]            # B x L, starts with BOS, padded with 0
with torch.no_grad():
    logits = transformer(x)              # B x L x V
pred   = logits[:, :-1, :]               # position i-1 predicts...
target = x[:, 1:]                        # ...token i
mask   = target != 0                     # ignore PAD
loss = torch.nn.functional.cross_entropy(pred[mask], target[mask])
print(loss.item(), loss.exp().item())

x = next(iter(val_loader))[0][:2].clone()
with torch.no_grad():
    a = transformer(x)
    x2 = x.clone(); x2[:, 60:] = 0       # blank out the future
    b = transformer(x2)
print((a[:, :59] - b[:, :59]).abs().max())   # must be ~0

# Final Matrix with probabilities for all next amino acids of FAM Batch_size*23
log_classifier = torch.zeros(BATCH_SIZE, VOCAB_SIZE) 

with torch.no_grad():
    for i in range(1, MAX_LENGTH):  # index runs from 1 to 128 inclusive
        print("Column nº",i)
        output_transformer = transformer(input)  # Re-feed next column
         
        for id_amino in range(3,len(AA)+3,1): 
            classifier_input = input.clone()
            classifier_input[:,i]= id_amino # Old sequence + current Amino acid
            logits_next_AA = classifier(classifier_input) # Batch_Size * len(FAMILIES) Probabilities family of current Amino acid string iteration
            log_next_AA = torch.log_softmax(logits_next_AA, dim=-1)
            log_classifier[:,id_amino] = log_next_AA[:,FAM] # Selection of guidence family 

        logits_transformer = output_transformer[:, i - 1, :] # Slice at current token prediction (-1 because we dont count the <BOS> column)
        # Combine probabilites of next token with probabilites of Amino beeing of FAM
        log_transformer = torch.log_softmax(logits_transformer, dim=-1)
        
        logits_guided = log_transformer + Lambda * log_classifier
        logits_guided[:, :2] = -float("inf")

        probabilities = torch.softmax(
            logits_guided, dim=1
        )  # Logits -> Probabilities
  
        input[:, i] = torch.multinomial(probabilities, num_samples=1).squeeze(
            1
        )  # Sample Column along all sequences and update input
print(input[0,:])



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
    "./results/sequences/guided_generated_sequences.csv", index=False
)
