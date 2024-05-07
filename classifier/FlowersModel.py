from torch import nn, Tensor


class FlowersModel(nn.Module):
    """
    Encapsulates a PyTorch Module intended to extract and learn the important features of the `Flowers-102` dataset from
    the VGG at Oxford University. See https://www.robots.ox.ac.uk/~vgg/data/flowers/102. This network is a feed-forward
    convolutional deep neural network using batch-normalisation.
    """
    def forward(self, data: Tensor) -> Tensor:
        """
        Specify the process of feeding data through the layers of the network

        :param data: The input data to propagate
        :return: The probabilistic predictions tensor from the model
        """
        data = self._pool(nn.functional.relu(self._bn1(self._conv1(data))))
        data = self._pool(nn.functional.relu(self._bn2(self._conv2(data))))
        data = self._pool(nn.functional.relu(self._bn3(self._conv3(data))))

        data = data.view(-1, 128 * 26 * 26)

        data = nn.functional.relu(self._lin1(data))
        data = nn.functional.relu(self._lin2(data))
        data = nn.functional.relu(self._lin3(data))

        return self._lin4(data)

    def __init__(self) -> None:
        """
        Configures and defines a convolutional neural network for the purposes of classifying the VGG `Flowers-102` data
        """

        super(FlowersModel, self).__init__()

        self._bn1 = nn.BatchNorm2d(32)
        self._bn2 = nn.BatchNorm2d(64)
        self._bn3 = nn.BatchNorm2d(128)

        self._conv1 = nn.Conv2d(3, 32, 3)
        self._conv2 = nn.Conv2d(32, 64, 3)
        self._conv3 = nn.Conv2d(64, 128, 3)

        self._pool = nn.MaxPool2d(2, 2)

        self._lin1 = nn.Linear(128 * 26 * 26, 2048)
        self._lin2 = nn.Linear(2048, 1024)
        self._lin3 = nn.Linear(1024, 512)
        self._lin4 = nn.Linear(512, 102)
