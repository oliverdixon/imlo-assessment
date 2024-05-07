import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Final

from FlowersModel import FlowersModel


class _ModelDriver:
    """
    Drives a PyTorch module/model by supporting a training and testing framework
    """
    _BATCH_SIZE: Final[int] = 16
    _SAVE_PATH = "data/flowers-net.pth"
    _FIGURES_DIRECTORY = "figures/"

    _DATASET_TEMPLATES: Final[dict[str, transforms.Compose]] = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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

    @staticmethod
    def _report_epoch_summary(training_accuracy: float, validation_accuracy: float, epoch_loss: float,
                              learning_rates: tuple[float, float]) -> None:
        """
        Reports the post-training epoch performance summary

        :param training_accuracy: The percentage accuracy of the training tests, over the entire epoch
        :param validation_accuracy: The percentage accuracy of the validation tests
        :param epoch_loss: The total loss values accumulated during the epoch
        :param learning_rates: The closing LR of the current epoch followed by the closing LR of the previous epoch
        """
        print(f"\t\tTraining data accumulated accuracy: {training_accuracy:.2f} %\n"
              f"\t\tValidation data accuracy: {validation_accuracy:.2f} %\n"
              f"\t\tClosing learning rate: {learning_rates[0]}"
              f"{' (scheduled)' if learning_rates[0] != learning_rates[1] else ''}\n"
              f"\t\tAccumulated training loss: {epoch_loss}")

    @staticmethod
    def _count_correct_predictions(probabilities: torch.Tensor, truths: torch.Tensor) -> int:
        """
        Given a tensor describing the category-wise probabilities over each input image in a batch, take the category
        reflecting the greatest confidence and count the number of correct model predictions.

        :param probabilities: Category-wise probabilities over each input image in a batch
        :param truths: Correct labels for each image in the batch
        :return: The number of correct predictions, based on the maximal probabilities
        """
        guesses = torch.max(probabilities, 1)[1]  # Collapse probabilities into a image-wise maximal labelling tensor
        return (guesses == truths).sum().item()

    def _evaluate_batch(self, batch_data: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a set of inputs against the current model

        :param batch_data: The input-label pairs for the current batch
        :return: The model outputs and the ground-truth labels
        """
        inputs = batch_data[0].to(torch.get_default_device())
        return self._model(inputs).to(torch.get_default_device()), batch_data[1].to(torch.get_default_device())

    def _validate_on_data(self, reference_data: DataLoader) -> float:
        """
        Evaluate the model over the entire given dataset

        :param reference_data: The data split on which the model should be tested
        :return: The percentage accuracy of the model
        """
        self._model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            data: list[torch.Tensor]
            for data in reference_data:
                outputs, labels = self._evaluate_batch(data)
                correct += self._count_correct_predictions(outputs, labels)
                total += labels.size(0)

        return correct / total * 100

    def train_model(self, epoch_count: int) -> None:
        """
        Trains the driven model on the driver instance's training data for the specified number of epochs, reporting the
        progress as the training proceeds

        :param epoch_count: The number of epochs for which the model should be trained
        """
        print("Training...")
        last_lr: float

        for epoch_idx in range(epoch_count):
            correct = 0
            total = 0
            epoch_loss = 0.0

            self._model.train()
            self._report_epoch(epoch_idx, epoch_count)

            data: list[torch.Tensor]
            for batch_idx, data in enumerate(self._training_data):
                self._optimiser.zero_grad()

                outputs, labels = self._evaluate_batch(data)
                loss = self._loss(outputs, labels)
                loss.backward()
                self._optimiser.step()

                total += labels.size(0)
                correct += self._count_correct_predictions(outputs, labels)
                epoch_loss += loss.item()

            last_lr = self._scheduler.get_last_lr()[0]
            self._scheduler.step(epoch_loss)
            self._report_epoch_summary(correct / total * 100, self._validate_on_data(self._validation_data), epoch_loss,
                                       (self._scheduler.get_last_lr()[0], last_lr))

        self._trained = True
        torch.save(self._model.state_dict(), _ModelDriver._SAVE_PATH)
        print(f"Finished training over {epoch_count} epochs: see '{_ModelDriver._SAVE_PATH}'.")

    def evaluate_model(self) -> None:
        """
        Evaluate the entire model against the set of training data and output the results
        """
        print("Testing...")
        if not self._trained:
            print("\tUntrained model; loading pre-trained model from " + _ModelDriver._SAVE_PATH)
            self._model.load_state_dict(torch.load(_ModelDriver._SAVE_PATH, map_location=torch.get_default_device()))

        print(f"\tEntire network accuracy: {self._validate_on_data(self._test_data):.2f} %.\n"
              f"Finished testing.")

    def graph_network(self) -> None:
        """
        Produce a vectorised PDF graph of the neural network layers and store in the FIGURES_DIRECTORY. In addition to
        TorchViz, this operation requires the GraphViz Python library and system program (verify with 'dot -V').
        """
        from torchviz import make_dot

        make_dot(self._model(next(iter(self._training_data))[0].to(torch.get_default_device())),
                 params=dict(list(self._model.named_parameters()))) \
            .render(_ModelDriver._FIGURES_DIRECTORY + "/network_graph", format="pdf", cleanup=True)

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initialise the model driver: prepare data in its train-test-validation split, and configure hyperparameters and
        model utility functions such as the optimiser, scheduler, and loss function.

        :param model: The model on which training, validation, and testing should be performed
        """
        self._model = model.to(torch.get_default_device())
        self._trained = False

        self._training_data = self._download_data_split("train")
        self._validation_data = self._download_data_split("val")
        self._test_data = self._download_data_split("test")

        self._loss = torch.nn.CrossEntropyLoss()

        # The optimiser uses a very large initial learning rate and allows the scheduler to correct it over the epochs.
        self._optimiser = torch.optim.SGD(self._model.parameters(), lr=0.001, momentum=0.9)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimiser, factor=0.2, patience=2)


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
    graph = False

    torch.set_default_device(identify_optimal_device())
    driver = _ModelDriver(FlowersModel())

    if graph:
        driver.graph_network()
    else:
        driver.train_model(100)
        driver.evaluate_model()
