from torch import nn, Tensor
from typing import Final


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
        # Apply the convolutions and batch normalisation
        for conv, bn in zip(self._convolutions, self._batchnorms):
            data = self._pool(self._activation(bn(conv(data))))

        # Collapse the convolved and pooled 128-channel tensors into a vector over each batch image
        data = data.view(-1, 128 * 26 * 26)

        # Linearise the data into the 102-component probability vector
        for lin in self._linears:
            data = self._activation(lin(data))

        return data

    def __init__(self) -> None:
        """
        Configures and defines a convolutional neural network for the purposes of classifying the VGG `Flowers-102` data
        """
        super(FlowersModel, self).__init__()

        self._batchnorms: Final[nn.ModuleList] = nn.ModuleList(list(map(nn.BatchNorm2d, [32, 64, 128])))
        self._convolutions: Final[nn.ModuleList] = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3) for
            in_channels, out_channels in [(3, 32), (32, 64), (64, 128)]
        ])
        self._linears: Final = nn.ModuleList([
            nn.Linear(in_features, out_features) for
            in_features, out_features in [(128 * 26 * 26, 2048), (2048, 1024), (1024, 512), (512, 102)]
        ])

        self._pool: Final[nn.MaxPool2d] = nn.MaxPool2d(2, 2)
        self._activation: Final[callable] = nn.functional.relu
