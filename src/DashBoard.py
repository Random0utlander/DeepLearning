
import sys
import time
import torch


class TrainingDashboard:

    def __init__(self, num_epochs):

        self.num_epochs = num_epochs
        self.start_time = time.perf_counter()
        self.lines_printed = 0

        # GPU information
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_available = True
        else:
            self.gpu_name = "CPU"
            self.gpu_available = False

    def update(
        self,
        epoch,
        batch,
        total_batches,
        train_loss,
        val_loss=None,
        status="TRAINING"
    ):

        # Elapsed time
        elapsed = time.perf_counter() - self.start_time

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Overall epoch progress
        progress = (
            (epoch - 1 + batch / total_batches)
            / self.num_epochs
        )

        progress = min(progress, 1.0)

        width = 30
        filled = int(width * progress)

        epoch_bar = (
            "█" * filled
            + "░" * (width - filled)
        )

        # Batch progress
        batch_progress = batch / total_batches
        batch_filled = int(width * batch_progress)

        batch_bar = (
            "█" * batch_filled
            + "░" * (width - batch_filled)
        )

        # GPU memory
        if self.gpu_available:

            allocated = (
                torch.cuda.memory_allocated(0) / 1024**3
            )

            total = (
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3
            )

            gpu_memory = f"{allocated:.2f} / {total:.2f} GB"

        else:
            gpu_memory = "N/A"

        # Loss formatting
        train_loss_str = (
            f"{train_loss:.4f}"
            if train_loss is not None
            else "--"
        )

        val_loss_str = (
            f"{val_loss:.4f}"
            if val_loss is not None
            else "--"
        )

        # Dashboard layout
        lines = [
            "=" * 60,
            "              PROTEIN TRANSFORMER",
            "=" * 60,
            f" STATUS: {status}",
            f" GPU: {self.gpu_name}",
            "",
            f" EPOCH: {epoch}/{self.num_epochs}",
            f" [{epoch_bar}] {progress*100:5.1f}%",
            "",
            f" BATCH: {batch}/{total_batches}",
            f" [{batch_bar}] {batch_progress*100:5.1f}%",
            "",
            f" TRAIN LOSS:      {train_loss_str}",
            f" VALIDATION LOSS: {val_loss_str}",
            "",
            f" ELAPSED TIME: {elapsed_str}",
            f" GPU MEMORY: {gpu_memory}",
            "=" * 60
        ]

        # Move cursor up and overwrite previous dashboard
        if self.lines_printed > 0:
            sys.stdout.write(f"\033[{self.lines_printed}A")
            sys.stdout.write("\033[J")  # Clear everything below cursor


        for line in lines:
            sys.stdout.write("\r\033[2K" + line + "\n")

        sys.stdout.flush()

        self.lines_printed = len(lines)
