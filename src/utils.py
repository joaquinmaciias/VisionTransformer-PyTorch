import torch
import numpy as np
import torch.nn.functional as F
from torch.jit import RecursiveScriptModule
import os
import random
from typing import Optional
import math


class CosineLR(torch.optim.lr_scheduler.LRScheduler):
    """
    Cosine learning-rate scheduler (single-cycle), similar to PyTorch's
    CosineAnnealingLR.

    It anneals each param group's LR following:
        lr_t = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t / T_max))

    Where:
      - t      is the current epoch counter (starting at 0 after the first step)
      - T_max  is the total number of scheduler steps for a full cosine cycle
      - eta_min is the minimum LR at the end of the cycle

    Attr:
        optimizer: the optimizer being scheduled.
        T_max: total number of steps (typically #epochs) in the cosine cycle.
        eta_min: minimum learning rate at the end of the cycle.
        last_epoch: last epoch index passed to .step() (starts at -1).
        base_lrs: initial LRs for each param group (captured at construction).
    """

    optimizer: torch.optim.Optimizer
    T_max: int
    eta_min: float
    last_epoch: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        """
        This method is the constructor of CosineLR class.

        Args:
            optimizer: optimizer to schedule.
            T_max: number of .step() calls to complete one cosine cycle
                   (commonly, the number of epochs).
            eta_min: floor (minimum) learning rate at the end of the cycle.
        """

        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.last_epoch = -1  # increments on first step() call to 0

        # Capture each param group's initial LR as base LR for the cosine curve
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def _lr_at(self, base_lr: float, t: int) -> float:
        """
        Compute the cosine-annealed LR for a given base_lr at step t.
        """

        # Clamp to avoid division by zero when T_max == 0
        T = max(self.T_max, 1)

        # Cosine formula (single cycle)
        return self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * t / T))

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Advance one step of the scheduler.

        Args:
            epoch: optional external epoch index. If provided, we use it as 't';
                   otherwise, we increment the internal counter.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            # Allow manual setting of the current "t"
            self.last_epoch = int(epoch)

        t = self.last_epoch

        # Update each param group's LR using its own base LR
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._lr_at(base_lr, t)


def get_dropout_random_indexes(shape: torch.Size, p: float) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # Build a ones tensor with the target shape
    inputs: torch.Tensor = torch.ones(shape)

    # Apply dropout to obtain the same pattern PyTorch would use internally
    dropped: torch.Tensor = F.dropout(inputs, p)

    # Convert to a mask of zeros where elements were dropped
    indexes = (dropped == 0).int()

    return indexes


class Accuracy:
    """
    Simple streaming accuracy.

    Attr:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: model outputs [batch, num_classes]
            labels: ground-truth class indices [batch]
        """

        # Predicted class per sample
        predictions = logits.argmax(1).type_as(labels)

        # Accumulate counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # Reset counts
        self.correct = 0
        self.total = 0

        return None


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # Create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # Define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # Set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # Set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
