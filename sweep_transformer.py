
import csv
import os
import gc
import torch
from itertools import product
from train_transformer import train_transformer

def run_sweep():

    # Hyperparameter search space
    embedding_dims = [32, 64, 128, 256]
    hidden_dims = [128, 256, 512, 1024]
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.0001, 0.0005, 0.001]

    experiments = list(product(
        embedding_dims,
        hidden_dims,
        batch_sizes,
        learning_rates
    ))

    print(f"Total experiments: {len(experiments)}")

    os.makedirs("results/transformer", exist_ok=True)

    results = []

    results_file = "results/transformer/sweep_results.csv"

    for i, (emb, hidden, batch, lr) in enumerate(experiments):

        print(f"\nExperiment {i+1}/{len(experiments)}")

        try:

            transformer, train_losses, val_losses, ckpt_path = (
                train_transformer(
                    dim_embedding=emb,
                    num_hidden=hidden,
                    batch_size=batch,
                    num_epochs=5,
                    learning_rate=lr
                )
            )

            best_loss = min(val_losses)
            best_epoch = val_losses.index(best_loss) + 1

            result = {
                "embedding": emb,
                "hidden": hidden,
                "batch_size": batch,
                "lr": lr,
                "best_val_loss": best_loss,
                "best_epoch": best_epoch
            }

            results.append(result)

            # Save results immediately
            file_exists = os.path.isfile(results_file)

            with open(results_file, "a", newline="") as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=result.keys()
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(result)

            print(f"Validation loss: {best_loss:.4f}")

            # Release GPU memory
            del transformer
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as error:

            print(f"Experiment failed: {error}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

    # Rank experiments
    results.sort(key=lambda x: x["best_val_loss"])

    print("\n========== TOP 10 CONFIGURATIONS ==========")

    for result in results[:10]:
        print(result)

    return results


if __name__ == "__main__":

    results = run_sweep()
