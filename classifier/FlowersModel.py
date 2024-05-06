import torch
from torch import nn


class FlowersModel(nn.Module):
    def forward(self, data):
        data = self._pool(nn.functional.relu(self._bn1(self._conv1(data))))
        data = self._pool(nn.functional.relu(self._bn2(self._conv2(data))))
        data = self._pool(nn.functional.relu(self._bn3(self._conv3(data))))

        data = data.view(-1, 128 * 26 * 26)

        data = nn.functional.relu(self._lin1(data))
        data = nn.functional.relu(self._lin2(data))
        data = nn.functional.relu(self._lin3(data))

        return self._lin4(data)

    def __init__(self):
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

        self._optimiser = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
