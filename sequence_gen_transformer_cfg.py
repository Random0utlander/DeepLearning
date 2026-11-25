import torch 
import pandas as pd
from torch import nn
import torch.nn.functional as F
from protgen.models.transformer_cfg import Transformer_cfg
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
import matplotlib.pyplot as plt

BATCH_SIZE = 100
MAX_LENGTH = 129

# Load the best transformer cfg model
best_transformer_cfg = torch.load("./src/protgen/models/transformer_cfg_best.pt", weights_only=True)

# i forgot to add vocab size to the saved model, so we need to calculate it here
VOCAB_SIZE = len(special_tokens) + len(AA) 

#Create the MLP model 
transformer_cfg = Transformer_cfg(
        vocab_size=VOCAB_SIZE,
        dim_embedding=best_transformer_cfg["dim_embedding"],
        num_hidden=best_transformer_cfg["hidden_units"],
        dim_head=best_transformer_cfg["head_dim"],
        dropout=best_transformer_cfg["dropout"],
        max_seq_len=MAX_LENGTH
    )

# Load the weights of the best model into the Transformer instance
transformer_cfg.load_state_dict(best_transformer_cfg["weights"])

# Set the model to evaluation mode
transformer_cfg.eval()



family_unconditional= torch.full((BATCH_SIZE,), 3, dtype=torch.long)
guidance_w= [0, 1, 2, 4] # introducing w=4 here introduced a bad result, just verify the bad results of a excessive guidance

with torch.no_grad():
    for family in [0, 1, 2]:
        family_conditional = torch.full((BATCH_SIZE,), family, dtype=torch.long)
    
        print(f"In family {family}")
        for w in guidance_w:
            input = torch.zeros(BATCH_SIZE, MAX_LENGTH, dtype=torch.long)
            input[:,0]=1

            for i in range(1, MAX_LENGTH):
                logits_cond = transformer_cfg(input[:,:i], family_conditional)
                logits_uncond = transformer_cfg(input[:, :i], family_unconditional)

                # log softmax for numeric stability and calibration
                log_prob_cond = torch.log_softmax(logits_cond[:, -1, :], dim=-1)
                log_prob_uncond = torch.log_softmax(logits_uncond[:, -1, :], dim=-1)

                log_prob_cfg = (1 + w)*log_prob_cond - w*log_prob_uncond

                probabilities = torch.softmax(log_prob_cfg, dim=-1)

                input[:, i] = torch.multinomial(probabilities, num_samples=1).squeeze(1)

            
            #Mask all the positions in front of the first <EOS> (2) inclusive
            mask = (input == 2).cumsum(dim=1) > 0
            output = input.masked_fill(mask, 0)


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

            sequences = [decode(seq) for seq in output]
            print(f"w={w}: {len(sequences)} generated sequences")
            pd.DataFrame(sequences, columns=["sequence"]).to_csv(f"./results/sequences/transformer_cfg_gen_seq_fam{family}_w{w}.csv", index=False)