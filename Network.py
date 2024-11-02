import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size: int):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.size(0) == 1:
            x = x.expand(2, -1)
            output = self.network(x)[:1]
            return output
        return self.network(x)