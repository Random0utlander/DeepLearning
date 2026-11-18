from train_mlp import train_mlp

dim_embedding = [16, 32, 64]
batch_size = [64, 128, 256, 512]
num_hidden = [64, 128, 256]
num_epochs = 10


# Sweep through the hyperparameters and find the best combination
def sweep_mlp(dim_embedding, num_hidden, batch_size, num_epochs):
    best_val_loss = float("inf")
    best_hyperparams = None
    for emb in dim_embedding:
        for hid in num_hidden:
            for batch in batch_size:
                print(f"Training with emb={emb}, hid={hid}, batch={batch}")
                mlp, train_losses, val_losses, ckpt_path = train_mlp(
                    dim_embedding=emb,
                    num_hidden=hid,
                    batch_size=batch,
                    num_epochs=num_epochs,
                )

                # Update best hyperparameters if current validation loss is lower
                if min(val_losses) < best_val_loss:
                    best_val_loss = min(val_losses)
                    best_hyperparams = (emb, hid, batch)

    return best_val_loss, best_hyperparams


# if __name__ == "__main__" makes sure that the code block is only executed
# when the script is run directly, not when imported as a module.
if __name__ == "__main__":
    best_val_loss, best_hyperparams = sweep_mlp(
        dim_embedding=dim_embedding,
        num_hidden=num_hidden,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    print(
        f"Best validation loss: {best_val_loss} with hyperparameters: {best_hyperparams}"
    )
