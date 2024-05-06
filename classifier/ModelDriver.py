import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Final

from FlowersModel import FlowersModel


class _ModelDriver:
    """
    Drives a PyTorch module/model by supporting a training and testing framework
    """
    _BATCH_SIZE: Final[int] = 4
    _BATCH_REPORT_RATE = 100
    _SAVE_PATH = "data/flowers_net.pth"

    _DATASET_TEMPLATES: Final[dict[str, transforms.Compose]] = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    @staticmethod
    def _report_epoch(epoch_idx: int, epoch_count: int) -> None:
        """
        Reports the current epoch progress

        :param epoch_idx: The current epoch
        :param epoch_count: The number of intended epochs
        """
        print(f"\tEpoch {epoch_idx} / {epoch_count - 1}")

    @staticmethod
    def _report_epoch_batch_set_performance(batch_idx: int, cumulative_loss: float) -> None:
        """
        Reports the training performance over a set of batches

        :param batch_idx: The index of the first batch appearing in the set
        :param cumulative_loss: The cumulative loss over the batch-set
        """
        print(f"\t\tBatches {(batch_idx - _ModelDriver._BATCH_REPORT_RATE + 1):03}--{batch_idx:03}: "
              f"Avg. loss: {cumulative_loss / _ModelDriver._BATCH_REPORT_RATE}")

    @staticmethod
    def _report_epoch_summary(training_accuracy: float, validation_accuracy: float) -> None:
        """
        Reports the post-training epoch performance summary

        :param training_accuracy: The percentage accuracy of the training tests, over the entire epoch
        :param validation_accuracy: The percentage accuracy of the validation tests
        """
        print(f"\t\tTraining data accuracy: {training_accuracy:.2f} %\n"
              f"\t\tValidation data accuracy: {validation_accuracy:.2f} %")

    @staticmethod
    def _download_data_split(split: str) -> DataLoader:
        """
        Downloads a particular split of the data and configures a PyTorch DataLoader with the fixed batch size.

        :param split: The identifier of the targeted split
        """
        return DataLoader(
            datasets.Flowers102(
                root="data",
                split=split,
                download=True,
                transform=_ModelDriver._DATASET_TEMPLATES[split]
            ),

            batch_size=_ModelDriver._BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator(torch.get_default_device())
        )

    def _evaluate_batch(self, batch_data: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a set of inputs against the current model

        :param batch_data: The input-label pairs for the current batch
        :return: The model outputs and the ground-truth labels
        """
        inputs = batch_data[0].to(torch.get_default_device())
        return self._model(inputs).to(torch.get_default_device()), batch_data[1].to(torch.get_default_device())

    def _validate_on_data(self, test_data: DataLoader) -> float:
        """
        Evaluate the model over the entire given dataset

        :param test_data: The data split on which the model should be tested
        :return: The percentage accuracy of the model
        """
        self._model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            data: list[torch.Tensor]
            for data in test_data:
                outputs, labels = self._evaluate_batch(data)
                predicted: torch.Tensor = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total * 100

    def train_model(self, epochs: int) -> None:
        """
        Trains the driven model on the driver instance's training data for the specified number of epochs, reporting the
        progress as the training proceeds

        :param epochs: The number of epochs for which the model should be trained
        """
        print("Training...")

        for epoch_idx in range(epochs):
            correct = 0
            total = 0

            self._model.train()
            self._report_epoch(epoch_idx, epochs)

            data: list[torch.Tensor]
            for batch_idx, data in enumerate(self._training_data):
                self._optimiser.zero_grad()

                outputs, labels = self._evaluate_batch(data)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optimiser.step()

                # TODO: wrap tensor-element-counting into a method
                _, predicted = torch.max(outputs, -1)
                correct += (predicted == labels).sum().item()
                total += predicted.size(0)

            self._report_epoch_summary(correct / total * 100, self._validate_on_data(self._validation_data))

        self._trained = True
        torch.save(self._model.state_dict(), _ModelDriver._SAVE_PATH)
        print(f"Finished training over {epochs} epochs.")

    def evaluate_model(self) -> None:
        """
        Evaluate the entire model against the set of training data and output the results
        """
        print("Testing...")

        if not self._trained:
            print("\tUntrained model; loading from " + _ModelDriver._SAVE_PATH)
            self._model.load_state_dict(torch.load(_ModelDriver._SAVE_PATH))

        print(f"\tEntire network accuracy: {self._validate_on_data(self._test_data):.2f} %.\n"
              f"Finished testing.")

    def __init__(self, model: torch.nn.Module):
        """
        Initialise the model driver: prepare data in its train-test-validation split, and configure hyperparameters and
        model utility functions such as the optimiser, scheduler, and loss function.

        :param model: The model on which training, validation, and testing should be performed
        """
        self._model = model.to(torch.get_default_device())
        self._trained = False

        self._training_data = self._download_data_split("train")
        self._test_data = self._download_data_split("test")
        self._validation_data = self._download_data_split("val")

        self._optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimiser) # TODO
        self._loss = torch.nn.CrossEntropyLoss()


def identify_optimal_device() -> str:
    """
    Selects the optimal single device (minimally utilised CUDA-equipped GPU or CPU, depending on CUDA support) on which
    model computations should be run. The final selection is reported, with a warning in the case of the CPU fallback.

    :return: The PyTorch-recognisable name of the optimal device
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("Warning: CUDA is not available; models will execute very slowly on the CPU.")
        return "cpu"

    name: str = "cuda:0"
    min_usage: int = torch.cuda.utilization(name)

    for i in range(1, torch.cuda.device_count()):
        if min_usage == 0:
            break

        candidate_name = "cuda:" + str(i)
        candidate_usage = torch.cuda.utilization(candidate_name)

        if candidate_usage <= min_usage:
            name = candidate_name
            min_usage = candidate_usage

    print(f"{torch.cuda.device_count()} CUDA GPUs are available; selecting \'{name}\' ({min_usage}% utilised).")
    return name


if __name__ == "__main__":
    torch.set_default_device(identify_optimal_device())
    driver = _ModelDriver(FlowersModel())

    driver.train_model(10)
    driver.evaluate_model()
