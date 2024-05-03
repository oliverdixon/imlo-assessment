import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Final


class FlowersModel(torch.nn.Module):
    """ Encapsulates a convolutional neural network model for the `Flowers-102` dataset from Oxford University; see
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.
    """
    __BATCH_SIZE: Final[int] = 64
    __DATASET_TEMPLATES: Final[dict[str, transforms.Compose]] = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),

        "test": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),

        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

    @staticmethod
    def __download_data_split(split: str) -> DataLoader:
        return DataLoader(
            datasets.Flowers102(
                root="data",
                split=split,
                download=True,
                transform=FlowersModel.__DATASET_TEMPLATES[split]
            ),

            batch_size=FlowersModel.__BATCH_SIZE
        )

    def __prepare_data(self) -> None:
        self.__training_data = self.__download_data_split("train")
        self.__test_data = self.__download_data_split("test")
        self.__validation_data = self.__download_data_split("val")

    def __init__(self, in_features, hidden_layers, out_features, activation=nn.functional.relu):
        super().__init__()
        self.__prepare_data()


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
    print(f"{torch.cuda.device_count()} CUDA GPUs are available; using {torch.get_default_device()}")

    classifier = FlowersModel(None, None, None)
