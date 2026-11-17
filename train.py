import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from protgen.models.mlp import MLP
from protgen.data import ProteinDataset
from protgen.data.make_tokens import TokenDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
import math
import matplotlib.pyplot as plt
import os

# Vocabulary size is the number of unique tokens in the dataset, which includes special tokens and amino acids
VOCAB_SIZE = len(special_tokens) + len(AA)
# mlp with knowledge of 3 previous tokens
WINDOW_SIZE = 3


# Load the datasets
train_data = ProteinDataset(path_to_data="./data/train.csv")
val_data = ProteinDataset(path_to_data="./data/val.csv")
test_data = ProteinDataset(path_to_data="./data/test.csv")

# Tokenize the datasets
train_dataset = TokenDataset(train_data, window_size=WINDOW_SIZE)
val_dataset = TokenDataset(val_data, window_size=WINDOW_SIZE)
test_dataset = TokenDataset(test_data, window_size=WINDOW_SIZE)


dim_embedding = 32
num_hidden = 128
batch_size = 512
num_epochs = 5

os.makedirs("checkpoints", exist_ok=True)


# Accuracy function
def accuracy(ys, ts):
    # Finding the token with the highest probability for each sample in the batch
    predictions = torch.argmax(ys, dim=1)
    # Comparing with the target and counting the number of correct predictions
    correct_prediction = (predictions == ts).sum().item()
    # Return counter of correct predictions
    return float(correct_prediction)


def train_mlp(dim_embedding, num_hidden, batch_size, num_epochs):

    torch.manual_seed(42)  # for reproducibility

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Define the checkpoint path for saving the model
    ckpt_path = (
        f"checkpoints/model_emb{dim_embedding}_hid{num_hidden}_batch{batch_size}.pt"
    )

    # store loss and accuracy for information
    train_losses, val_losses, train_accs, val_accs = (
        [],
        [],
        [],
        [],
    )

    # Create the MLP model
    mlp = MLP(
        vocab_size=VOCAB_SIZE,
        dim_embedding=dim_embedding,
        num_hidden=num_hidden,
        window_size=WINDOW_SIZE,
    )

    # Optimizer function
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    # training loop
    best_val_loss = float("inf")
    best_epoch = 0
    for e in range(num_epochs):

        # -----------------TRAINING-----------------------#
        mlp.train()  # set model to training mode

        batch_train_loss, batch_train_acc, batch_count, num_examples = 0, 0, 0, 0
        for x_train_batch, targets_train_batch in train_loader:
            # zeroize accumulated gradients in parameters
            optimizer.zero_grad()
            # predict by running forward pass
            tr_output = mlp(x_train_batch)
            # compute cross entropy loss
            tr_loss = F.cross_entropy(
                tr_output, targets_train_batch, ignore_index=IGNORE_INDEX
            )
            # compute gradients given loss
            tr_loss.backward()
            # update the parameters given the computed gradients
            optimizer.step()
            batch_train_acc += accuracy(tr_output, targets_train_batch)

            # store training batch loss
            # Cross entropy loss is averaged over the batch, so we multiply by the batch size to get the total loss for the batch
            batch_train_loss += tr_loss.item() * x_train_batch.size(0)
            batch_count += 1
            num_examples += x_train_batch.size(0)

        # store average training loss and accuracy for the epoch
        train_losses.append(batch_train_loss / (num_examples))
        train_accs.append(batch_train_acc / (num_examples))

        # -----------------VALIDATION-----------------------#

        mlp.eval()  # set model to evaluation mode

        with torch.no_grad():  # disable gradient computation for validation
            batch_val_loss, batch_val_acc, batch_count, num_examples = 0, 0, 0, 0
            for x_val_batch, targets_val_batch in val_loader:
                # predict with validation input
                val_output = mlp(x_val_batch)
                # compute loss and accuracy
                batch_val_loss += F.cross_entropy(
                    val_output, targets_val_batch, ignore_index=IGNORE_INDEX
                ).item() * x_val_batch.size(0)
                batch_val_acc += accuracy(val_output, targets_val_batch)
                batch_count += 1
                num_examples += x_val_batch.size(0)

            # store loss and accuracy
            val_losses.append(batch_val_loss / (num_examples))
            val_accs.append(batch_val_acc / (num_examples))

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_epoch = e
                torch.save(
                    {
                        "weights": mlp.state_dict(),
                        "embedding_size": dim_embedding,
                        "hidden_units": num_hidden,
                        "window_size": WINDOW_SIZE,
                        "epoch": best_epoch,
                        "loss": best_val_loss,
                    },
                    ckpt_path,
                )

        if e % 1 == 0:
            print(
                "Epoch %i, "
                "Train Cost: %0.3f"
                "\t Train Acc: %0.3f"
                "\tVal Cost: %0.3f"
                "\t Val acc: %0.3f"
                % (e, train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1])
            )

    print(f"Best validation loss: {best_val_loss:.3f} at epoch {best_epoch}")

    ckpt = torch.load(ckpt_path, weights_only=True)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} with loss {ckpt['loss']:.3f}")
    mlp.load_state_dict(ckpt["weights"])

    return mlp, train_losses, val_losses, ckpt_path


test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


def test_mlp(ckpt_path):
    test_acc, test_loss = 0, 0

    ckpt = torch.load(ckpt_path, weights_only=True)

    mlp = MLP(
        vocab_size=VOCAB_SIZE,
        dim_embedding=ckpt["embedding_size"],
        num_hidden=ckpt["hidden_units"],
        window_size=ckpt["window_size"],
    )

    mlp.load_state_dict(ckpt["weights"])

    # -----------------TESTING-----------------------#
    mlp.eval()  # set model to evaluation mode
    with torch.no_grad():  # disable gradient computation for validation
        batch_test_loss, batch_test_acc, batch_count, num_examples = 0, 0, 0, 0
        print("running test")
        for x_test_batch, targets_test_batch in test_loader:

            test_output = mlp(x_test_batch)
            # store this later

            # compute loss and accuracy
            batch_test_loss += F.cross_entropy(
                test_output, targets_test_batch, ignore_index=IGNORE_INDEX
            ).item() * x_test_batch.size(0)

            batch_test_acc += accuracy(test_output, targets_test_batch)
            num_examples += x_test_batch.size(0)
    # store loss and accuracy

    test_loss = batch_test_loss / (num_examples)
    test_acc = batch_test_acc / (num_examples)
    test_perplexity = math.exp(test_loss)

    print(
        "Test Cost: %0.3f"
        "\t Test acc: %0.3f"
        "\t Test perplexity: %0.3f" % (test_loss, test_acc, test_perplexity)
    )

    return test_loss, test_acc, test_perplexity


# if name is used to check if the script is being run directly (as opposed to being imported as a module).
# If it is, the code inside this block will be executed.
# This is a common Python idiom for making code both importable and executable.
if __name__ == "__main__":
    mlp, train_losses, val_losses, ckpt_path = train_mlp(
        dim_embedding=dim_embedding,
        num_hidden=num_hidden,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    test_loss, test_acc, test_perplexity = test_mlp(ckpt_path)

    # -----------------ENTROPY-----------------------#
    counts = torch.bincount(train_dataset.targets)
    probability_token = counts.float() / counts.sum()
    probability_token = probability_token[
        probability_token > 0
    ]  # filter out zero probabilities
    entropy = -torch.sum(
        probability_token * torch.log(probability_token)
    )  # torch log has natural log just like cross entropy loss

    # print("Token probabilities in the training set:")
    # for i, p in enumerate(probability_token):
    #    print(f"Token {i}: {p:.3f}")
    # print(f"Entropy: {entropy:.3f}")

    # -----------------PLOT-----------------------#
    plt.plot(range(0, num_epochs), train_losses, label="Train Loss")
    plt.plot(range(0, num_epochs), val_losses, label="Validation Loss")
    # plt.plot(entropy * torch.ones(num_epochs), label="Entropy", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("results/mlp_loss_curves.png")
