import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from src.models import VisionTransformer
from src.data import load_data
from src.utils import (
    Accuracy,
    save_model,
    set_seed,
)
import os

# Set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# Set all seeds and set number of threads
set_seed(45)

# Gloabl variable
DATA_PATH: str = "data"
NUMBER_OF_CLASSES: int = 10


def main() -> None:
    """
    This function is the main program for the training.
    """

    # Hyperparameters
    lr = 0.001
    epochs = 30
    log_interval = 100
    batch_size = 32
    weight_decay = 0.0001

    # Model hyperparameters
    sequence_length = 32
    d_model = 1024
    n_heads = 16

    # Load the data
    train_loader, valid_loader, test_loader = load_data(
        path=DATA_PATH, batch_size=batch_size, n_patches=sequence_length
    )
    print(f"Data loaded")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Valid size: {len(valid_loader.dataset)}\n")

    patch_size = train_loader.dataset[0][0].shape[1]
    print(f"Patch size: {patch_size}\n")
    print(f"N patches: {train_loader.dataset[0][0].shape[0]}\n")

    # Create the model
    model = VisionTransformer(
        sequence_length=sequence_length,
        patch_size=patch_size,
        d_model=d_model,
        number_of_heads=n_heads,
        number_of_classes=NUMBER_OF_CLASSES,
    ).to(device)
    print(model)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Create the accuracy
    train_accuracy = Accuracy()
    valid_accuracy = Accuracy()

    # Create the tensorboard writer
    writer = SummaryWriter()

    # Train the model
    for epoch in tqdm(range(1, epochs + 1)):
        # Train the model
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_accuracy.update(output, target)

            if batch_idx % log_interval == 0:
                tqdm.write(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

            writer.add_scalar("Loss/train", loss.item(), epoch)

        scheduler.step()
            
        # Validate the model
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_accuracy.update(output, target)

        # Log the accuracy
        writer.add_scalar("Accuracy/train", train_accuracy.compute(), epoch)
        writer.add_scalar("Accuracy/valid", valid_accuracy.compute(), epoch)

        tqdm.write(
            f"Epoch: {epoch} | "
            f"Train accuracy: {train_accuracy.compute():.4f} | "
            f"Valid accuracy: {valid_accuracy.compute():.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Reset the accuracy
        train_accuracy.reset()
        valid_accuracy.reset()

    # Test the model
    model.eval()
    test_accuracy = Accuracy()
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_accuracy.update(output, target)
            loss += criterion(output, target).item()

    print(f"Test accuracy: {test_accuracy.compute()}")
    print(f"Test loss: {loss / len(test_loader)}")

    # Save the model
    i = len(os.listdir("models")) + 1

    df = pd.read_csv("metrics.csv")
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "model": f"model_{i}",
                    "test_accuracy": test_accuracy.compute(),
                    "test_loss": loss / len(test_loader),
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "sequence_length": sequence_length,
                    "d_model": d_model,
                    "n_heads": n_heads,
                },
                index=[0],
            ),
        ]
    )
    df.to_csv("metrics.csv", index=False)

    save_model(model, f"model_{i}")


if __name__ == "__main__":
    main()
