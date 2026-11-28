import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from protgen.models.mlp import MLP
from protgen.data import ProteinDataset
from protgen.data.make_tokens import TokenDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens
from protgen.models.classifier_guided import Classifier
from protgen.models.transformer import Transformer

NUM_EPOCHS = 5

# Load the datasets
train_data = ProteinDataset(path_to_data="./data/train.csv")
val_data = ProteinDataset(path_to_data="./data/val.csv")
test_data = ProteinDataset(path_to_data="./data/test.csv")


# Load the best MLP model
best_transformer = torch.load(
    "./src/protgen/models/transformer_best.pt", weights_only=True
)

# Create the MLP model
transformer = Transformer(
    dim_embedding=best_transformer["dim_embedding"],
    num_hidden=best_transformer["hidden_units"],
    dim_head=best_transformer["head_dim"],
    dropout=best_transformer["dropout"],
)

transformer.load_state_dict(best_transformer["weights"])  # <- the missing line

classifier = Classifier(transformer, num_features=best_transformer["dim_embedding"])

# Optimizer function
optimizer = optim.Adam(classifier.parameters(), lr=0.001)


# Accuracy function
def accuracy(x, y):
    # Finding the token with the highest probability for each sample in the batch
    predictions = torch.argmax(x, dim=1)
    # Comparing with the target and counting the number of correct predictions
    correct_prediction = (predictions == y).sum().item()
    # Return counter of correct predictions
    return float(correct_prediction)


"""
#Tokenize the datasets
train_dataset = TokenDataset(train_data, window_size=WINDOW_SIZE)
val_dataset = TokenDataset(val_data, window_size=WINDOW_SIZE)
test_dataset = TokenDataset(test_data, window_size=WINDOW_SIZE)
"""


# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
val_loader = DataLoader(val_data, batch_size=512, shuffle=False)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False)


# store loss and accuracy for information
train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = (
    [],
    [],
    [],
    [],
    [],
    [],
)


print(f"Training for {NUM_EPOCHS} epochs...")
# training loop
for e in range(NUM_EPOCHS):

    # -----------------TRAINING-----------------------#
    classifier.train()  # set model to training mode

    batch_train_loss, batch_train_acc, batch_count, num_examples = 0, 0, 0, 0

    for x_train_batch, _, y, _ in train_loader:
        # zeroize accumulated gradients in parameters
        optimizer.zero_grad()
        # predict by running forward pass
        tr_output = classifier(x_train_batch)
        # compute cross entropy loss
        tr_loss = F.cross_entropy(tr_output, y, ignore_index=IGNORE_INDEX)
        # compute gradients given loss
        tr_loss.backward()
        # update the parameters given the computed gradients
        optimizer.step()
        batch_train_acc += accuracy(tr_output, y)

        # store training batch loss
        # Cross entropy loss is averaged over the batch, so we multiply by the batch size to get the total loss for the batch
        batch_train_loss += tr_loss.item() * x_train_batch.size(0)
        batch_count += 1
        num_examples += x_train_batch.size(0)

    # store average training loss and accuracy for the epoch
    train_losses.append(batch_train_loss / (num_examples))
    train_accs.append(batch_train_acc / (num_examples))

    # -----------------VALIDATION-----------------------#

    classifier.eval()  # set model to evaluation mode

    with torch.no_grad():  # disable gradient computation for validation
        batch_val_loss, batch_val_acc, batch_count, num_examples = 0, 0, 0, 0
        for x_val_batch, _, y, _ in val_loader:
            # predict with validation input
            val_output = classifier(x_val_batch)
            # compute loss and accuracy
            batch_val_loss += F.cross_entropy(
                val_output, y, ignore_index=IGNORE_INDEX
            ).item() * x_val_batch.size(0)
            batch_val_acc += accuracy(val_output, y)
            batch_count += 1
            num_examples += x_val_batch.size(0)

    # store loss and accuracy
    val_losses.append(batch_val_loss / (num_examples))
    val_accs.append(batch_val_acc / (num_examples))

    if e % 1 == 0:
        print(
            "Epoch %i, "
            "Train Cost: %0.3f"
            "\t Train Acc: %0.3f"
            "\tVal Cost: %0.3f"
            "\t Val acc: %0.3f"
            % (e, train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1])
        )

# Save the model?
torch.save(classifier.state_dict(), "./src/protgen/models/classifier.pt")


with torch.no_grad():  # disable gradient computation for validation
    batch_test_loss, batch_test_acc, batch_count, num_examples = 0, 0, 0, 0
    for x_test_batch, _, y, _ in test_loader:

        test_output = classifier(x_test_batch)

        # store this later

        # compute loss and accuracy
        batch_test_loss += F.cross_entropy(
            test_output, y, ignore_index=IGNORE_INDEX
        ).item() * x_test_batch.size(0)

        batch_test_acc += accuracy(test_output, y)
        batch_count += 1
        num_examples += x_test_batch.size(0)

# store loss and accuracy
test_losses.append(batch_test_loss / (num_examples))
test_accs.append(batch_test_acc / (num_examples))
