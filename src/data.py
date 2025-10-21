import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import requests
import tarfile
import shutil
from requests.models import Response
from tarfile import TarFile
from PIL import Image


class ImagenetteDataset(Dataset):
    """
    This class is the Imagenette Dataset.
    """

    def __init__(self, path: str, n_patches: int = 10) -> None:
        """
        Constructor of ImagenetteDataset.

        Args:
            path: path of the dataset.
            n_patches: number of patches to extract from the image.
        """

        # Set attributes
        self.path = path
        self.names = os.listdir(path)
        self.n_patches = n_patches

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
            [n_patches, P^2*C], where P is the resolution of the
            patch and C is the number of channels.
        """

        # Load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # Load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        image = image.view(self.n_patches, -1)

        return image, label


def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0, n_patches: int = 8
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagenette dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """

    # Download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # Create main dir
        os.makedirs(f"{path}")

        # Download data
        download_data(path)

    # Create datasets
    train_dataset: Dataset = ImagenetteDataset(f"{path}/train", n_patches)
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(train_dataset, [0.85, 0.15])
    test_dataset: Dataset = ImagenetteDataset(f"{path}/val", n_patches)

    # Define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


def download_data(path: str) -> None:
    """
    This function downloads the data from internet.

    Args:
        path: path to dave the data.
    """

    # Define paths
    url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    target_path: str = f"{path}/imagenette2.tgz"

    # Download tar file
    response: Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    # Extract tar file
    tar_file: TarFile = tarfile.open(target_path)
    tar_file.extractall(path)
    tar_file.close()

    # Create final save directories
    os.makedirs(f"{path}/train")
    os.makedirs(f"{path}/val")

    # Define resize transformation
    transform = transforms.Resize((224, 224))

    # Loop for saving processed data
    list_splits: tuple[str, str] = ("train", "val")
    for i in range(len(list_splits)):
        list_class_dirs = os.listdir(f"{path}/imagenette2/{list_splits[i]}")
        for j in range(len(list_class_dirs)):
            list_dirs = os.listdir(
                f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}"
            )
            for k in range(len(list_dirs)):
                image = Image.open(
                    f"{path}/imagenette2/{list_splits[i]}/"
                    f"{list_class_dirs[j]}/{list_dirs[k]}"
                )
                image = transform(image)
                if image.im.bands == 3:
                    image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

    # Delete other files
    os.remove(target_path)
    shutil.rmtree(f"{path}/imagenette2")

    return None


if __name__ == "__main__":
    load_data("data")
