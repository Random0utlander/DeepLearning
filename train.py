import torch 
from torch import nn
import torch.nn.functional as F
from torch import optim
from protgen.models.mlp import MLP
from protgen.data import ProteinDataset
from protgen.data.make_tokens import TokenDataset
from torch.utils.data import DataLoader
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens


#Variables for the model
VOCAB_SIZE = len(special_tokens) + len(AA)
DIM_EMBEDDING = 32
NUM_HIDDEN = 128
WINDOW_SIZE = 3

#Load the datasets
train_data = ProteinDataset(path_to_data="./data/train.csv")
val_data = ProteinDataset(path_to_data="./data/val.csv")
test_data = ProteinDataset(path_to_data="./data/test.csv")

#Create the MLP model 
mlp = MLP(
        vocab_size=VOCAB_SIZE,
        dim_embedding=DIM_EMBEDDING,
        num_hidden=NUM_HIDDEN,
        window_size=WINDOW_SIZE
    )

#Optimizer function
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

#Accuracy function
def accuracy(ys, ts):
    #Finding the token with the highest probability for each sample in the batch
    predictions = torch.argmax(ys, dim=1)
    #Comparing with the target and counting the number of correct predictions
    correct_prediction = (predictions == ts).sum().item()
    # Return counter of correct predictions 
    return float(correct_prediction)

#Tokenize the datasets
train_dataset = TokenDataset(train_data, window_size=WINDOW_SIZE)
val_dataset = TokenDataset(val_data, window_size=WINDOW_SIZE)
test_dataset = TokenDataset(test_data, window_size=WINDOW_SIZE)


#Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


# number of training passses
num_epochs = 5
# store loss and accuracy for information
train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = [], [], [], [], [], []


print(f"Training for {num_epochs} epochs...\n Starting Loss should be around ln(23) ≈ 3.1 for the first epoch")
# training loop
for e in range(num_epochs):
    
    #-----------------TRAINING-----------------------#

    print("here")
    mlp.train()  # set model to training mode

    batch_train_loss, batch_train_acc, batch_count, num_examples = 0, 0, 0, 0
    for x_train_batch, targets_train_batch in train_loader:
        # zeroize accumulated gradients in parameters
        optimizer.zero_grad()
        # predict by running forward pass
        tr_output = mlp(x_train_batch)
        # compute cross entropy loss
        tr_loss = F.cross_entropy(tr_output, targets_train_batch, ignore_index=IGNORE_INDEX)
        # compute gradients given loss
        tr_loss.backward()
        # update the parameters given the computed gradients
        optimizer.step()
        batch_train_acc += accuracy(tr_output, targets_train_batch)

        # store training batch loss
        # Cross entropy loss is averaged over the batch, so we multiply by the batch size to get the total loss for the batch
        batch_train_loss += tr_loss.item()*x_train_batch.size(0)
        batch_count += 1
        num_examples += x_train_batch.size(0)

    # store average training loss and accuracy for the epoch
    train_losses.append(batch_train_loss / (num_examples))
    train_accs.append(batch_train_acc / (num_examples))

    #-----------------VALIDATION-----------------------#

    mlp.eval()  # set model to evaluation mode
    
    with torch.no_grad():  # disable gradient computation for validation
        batch_val_loss, batch_val_acc, batch_count, num_examples = 0, 0, 0, 0
        for x_val_batch, targets_val_batch in val_loader:    
            # predict with validation input
            val_output = mlp(x_val_batch)
            # compute loss and accuracy
            batch_val_loss += F.cross_entropy(val_output, targets_val_batch, ignore_index=IGNORE_INDEX).item()*x_val_batch.size(0)
            batch_val_acc += accuracy(val_output, targets_val_batch)
            batch_count += 1
            num_examples += x_val_batch.size(0)

    # store loss and accuracy
    val_losses.append(batch_val_loss / (num_examples))
    val_accs.append(batch_val_acc / (num_examples))

    if e % 1 == 0:
        print("Epoch %i, "
              "Train Cost: %0.3f"
              "\t Train Acc: %0.3f"
              "\tVal Cost: %0.3f"
              "\t Val acc: %0.3f" % (e,
                                     train_losses[-1],
                                     train_accs[-1],
                                     val_losses[-1],
                                     val_accs[-1]))
    
#Save the model?
torch.save(mlp.state_dict(), "./state_dict.pt")


for e in range(num_epochs) :
     #-----------------TESTING-----------------------#
     
         #mlp.test()  # set model to test mode
         
         with torch.no_grad():  # disable gradient computation for validation
             batch_test_loss, batch_test_acc, batch_count, num_examples = 0, 0, 0, 0
             for x_test_batch, targets_test_batch in test_loader:    
                 
                 test_output = mlp(x_test_batch)

                #store this later 

                 # compute loss and accuracy
                 batch_test_loss += F.cross_entropy(test_output, targets_val_batch, ignore_index=IGNORE_INDEX).item()*x_val_batch.size(0)

                 batch_test_acc += accuracy(test_output, targets_test_batch)
                 batch_count += 1
                 num_examples += x_test_batch.size(0)
     
         # store loss and accuracy
         test_losses.append(batch_test_loss / (num_examples))
         test_accs.append(batch_val_acc / (num_examples))


