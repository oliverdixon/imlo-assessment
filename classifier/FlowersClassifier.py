import torch


class FlowersModel(torch.nn.Module):
    def __init__(self, in_features, hidden_layers, out_features, activation=torch.nn.functional.relu):
        super().__init__()
        print("Hello, world!")
        print(torch.cuda.is_available())


if __name__ == '__main__':
    classifier = FlowersModel(None, None, None)
