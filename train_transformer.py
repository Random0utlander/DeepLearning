import torch 
from torch import nn
from torch import optim
from protgen.models.transformer import Transformer
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
import math 
import os
import matplotlib.pyplot as plt
from protgen.data.make_data import IGNORE_INDEX
from torch.optim.lr_scheduler import LambdaLR
from src.DashBoard import TrainingDashboard

#Hyperparameters
DIM_EMBEDDING = 256
NUM_HIDDEN = 1024
BATCH_SIZE = 16
NUM_EPOCHS = 200
DIM_HEAD = 16
DROPOUT= 0.1

#Load the datasets
train_data = ProteinDataset(path_to_data="./data/train.csv")
val_data = ProteinDataset(path_to_data="./data/val.csv")
test_data = ProteinDataset(path_to_data="./data/test.csv")

# creation of checkpoints directory if it doesn't exist
os.makedirs("checkpoints/transformer", exist_ok=True)

# From Transformer paper "Atention is all you need" eq.3 for varing learning rate
def noam(d_model, warmup):
    def f(step):
        step = max(step, 1)
        return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
    return f

def train_transformer(dim_embedding, num_hidden, batch_size,dropout=DROPOUT, dim_head=DIM_HEAD, num_epochs=20,seed =67, best=False):

    torch.manual_seed(seed) # for reproducibility ;)

    dashboard = TrainingDashboard(num_epochs)

   
    # ---------------- DEVICE ---------------- #
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        
    print(f"Training on: {device}")
    print(f"Is bf16 suported? {torch.cuda.is_bf16_supported()}")

    # DataLoaders
     # ---------------- DATALOADERS ---------------- #
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    
    # Define the checkpoint path for saving the model; 
    ckpt_path = (
        f"checkpoints/transformer/"
        f"emb{dim_embedding}_hid{num_hidden}_"
        f"batch{batch_size}_drp{dropout}_seed{seed}.pt"
    )
    if best : ckpt_path = "./src/protgen/models/transformer_best.pt"
    # store loss and accuracy for information
    train_losses, val_losses, train_accs, val_accs = (
            [],
            [],
            [],
            [],
        )
    
    #Initialize transformer model 
    transformer= Transformer(
        dim_embedding=dim_embedding,
        num_hidden=num_hidden,
        dropout=dropout,
        dim_head=dim_head
        ).to(device)

    #Loss function
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1) #ignore padding tokens


    # Warmup 
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.05 * total_steps)   # 5-10% of total
    print(f"Steps per epoch: {steps_per_epoch} \nTotal steps: {total_steps} \nWarmup steps (5% of Total steps): {warmup_steps} " )
    
    # betas and eps from "Atention is all you need"
    optimizer = optim.AdamW(transformer.parameters(), lr=1.0,
                       betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, noam(dim_embedding, warmup=warmup_steps))
    
    #Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    for e in range(num_epochs):

        # -----------------TRAINING-----------------------#
        transformer.train()

        # Dashboard Update rate
        update_every = max(1, len(train_loader) // 50)


        train_loss_accumulated, train_token_count, train_accuracy, num_tokens = 0.0, 0, 0, 0
        for batch_idx,( x, x_shifted, _ , _) in enumerate(train_loader):

            x = x.to(device, non_blocking=True)
            x_shifted = x_shifted.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype= torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            ):

                optimizer.zero_grad()
                logits= transformer(x) # (batch of proteins, max_seq_len, vocab_size)
                logits_reshaped= logits.reshape(-1, logits.size(-1)) # (batch of proteins * max_seq_len, vocab_size)

                targets= x_shifted.reshape(-1)
                loss= cross_entropy_loss(logits_reshaped, targets)

                loss.backward()

            # Optimizer step followed by learning rate update
            optimizer.step()
            scheduler.step()

            valid_tokens = (
                    targets != IGNORE_INDEX
                ).sum().item()

            train_loss_accumulated += loss.item() * valid_tokens
            train_token_count += valid_tokens

            #Dashboard update 
            if (batch_idx + 1) % update_every == 0 or batch_idx + 1 == len(train_loader):
                dashboard.update(
                    epoch=e + 1,
                    batch=batch_idx + 1,
                    total_batches=len(train_loader),
                    train_loss=train_loss_accumulated / train_token_count,
                    val_loss=val_losses[-1] if val_losses else None,
                    status="TRAINING"
                )


        train_losses.append(train_loss_accumulated / train_token_count)

            
        # -----------------VALIDATION-----------------------#
        transformer.eval()

        with torch.no_grad():
            val_loss_accumulated, train_token_count, val_accuracy, num_tokens = 0.0, 0, 0, 0
            for x, x_shifted, _ , _ in val_loader:
                x = x.to(device, non_blocking=True)
                x_shifted = x_shifted.to(device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                ):
                    logits= transformer(x) # (batch of proteins, max_seq_len, vocab_size)
                    logits_reshaped= logits.reshape(-1, logits.size(-1)) # (batch of proteins * max_seq_len, vocab_size)

                    targets= x_shifted.reshape(-1)
                    loss= cross_entropy_loss(logits_reshaped, targets)
                    valid_tokens = (
                            targets != IGNORE_INDEX
                        ).sum().item()

                    val_loss_accumulated += loss.item()* valid_tokens
                    train_token_count += valid_tokens

        
            val_losses.append(val_loss_accumulated / train_token_count)


        #Saving checkpoints until finding best hyperparameters
        if val_losses[-1] < best_val_loss:
                        best_val_loss = val_losses[-1]
                        best_epoch = e + 1 
                        torch.save(
                            {
                                "weights": transformer.state_dict(),
                                "dim_embedding": dim_embedding,
                                "hidden_units": num_hidden,
                                "epoch": best_epoch,
                                "dropout": dropout,
                                "loss": best_val_loss,
                                "head_dim": dim_head
                            },
                            ckpt_path,
                        )
        """print(
            f"Epoch {e+1}/{num_epochs} | "
            f"Train loss: {train_losses[-1]:.4f} | "
            f"Val loss: {val_losses[-1]:.4f}"
        )"""


    print(f"Best validation loss: {best_val_loss:.3f} at epoch {best_epoch}")
    
     # ---------------- LOAD BEST MODEL ---------------- #
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True
    )

    transformer.load_state_dict(ckpt["weights"])

    return transformer, train_losses, val_losses, ckpt_path

def test_transformer(ckpt_path):

    #Create DataLoader
    test_loader = DataLoader(test_data, batch_size=512,shuffle=False)

    #Load best stat_dict
    ckpt = torch.load(ckpt_path, weights_only=True)

    #Initialize Transformer
    transformer = Transformer(
          dim_embedding=ckpt["dim_embedding"],
          num_hidden=ckpt["hidden_units"]
    )
    
    transformer.load_state_dict(ckpt["weights"])

    #Loss function
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1) #ignore padding tokens
    
    
    test_acc, test_loss = 0, 0
    # -----------------TESTING-----------------------#
    transformer.eval()
    with torch.no_grad():
        test_loss_accumulated, num_batches, test_accuracy, num_tokens = 0.0, 0, 0, 0
        for x, x_shifted, _ , _ in test_loader:
            logits= transformer(x) # (batch of proteins, max_seq_len, vocab_size)
            logits_reshaped= logits.reshape(-1, logits.size(-1)) # (batch of proteins * max_seq_len, vocab_size)

            targets= x_shifted.reshape(-1)
            loss= cross_entropy_loss(logits_reshaped, targets)


            #same as train, necessary to exclude padding
            targets_masked = targets[targets != -1]
            logits_masked = logits_reshaped[targets != -1]
            correct_tokens = (
                    logits_masked.argmax(dim=-1) == targets_masked
                            ).sum().item()

            test_accuracy += correct_tokens

            num_tokens += targets_masked.size(0)

            test_loss_accumulated += loss.item()*targets_masked.size(0)

        test_loss = test_loss_accumulated / num_tokens
        test_acc = test_accuracy / num_tokens
        test_perplexity = math.exp(test_loss)

    print(
            "Test Cost: %0.3f"
            "\t Test acc: %0.3f"
            "\t Test perplexity: %0.3f" % (test_loss, test_acc, test_perplexity)
        )
    
    return test_loss, test_acc, test_perplexity

if __name__ == "__main__":
    
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_bf16_supported())
    
    transformer, train_losses, val_losses, ckpt_path = train_transformer(
        dim_embedding=DIM_EMBEDDING,
        num_hidden=NUM_HIDDEN,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        best=True,
        dropout=DROPOUT,
        dim_head=DIM_HEAD
    )
    test_loss, test_acc, test_perplexity = test_transformer(ckpt_path)

    # -----------------ENTROPY-----------------------#
    mask = (train_data.tokens_shifted != IGNORE_INDEX)
    counts = torch.bincount(train_data.tokens_shifted[mask])
    probability_token = counts.float() / counts.sum()
    probability_token = probability_token[
        probability_token > 0
    ]  # filter out zero probabilities
    entropy = -torch.sum(
        probability_token * torch.log(probability_token)
    )  # torch log has natural log just like cross entropy loss

    print(f"Entropy: {entropy:.3f}")

    # -----------------PLOT-----------------------#
    plt.plot(range(0, NUM_EPOCHS), train_losses, label="Train Loss")
    plt.plot(range(0, NUM_EPOCHS), val_losses, label="Validation Loss")
    # plt.plot(entropy * torch.ones(num_epochs), label="Entropy", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("results/transformer/transformer_loss_curves.png")