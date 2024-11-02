import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size: int):
        super(Network, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )
        
        self.reveal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
        
        self.flag_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.size(0) == 1:
            x = x.expand(2, -1)
            shared_features = self.shared_layers(x)
            reveal_output = self.reveal_head(shared_features)
            flag_output = self.flag_head(shared_features)
            return reveal_output[:1], flag_output[:1]
        
        shared_features = self.shared_layers(x)
        reveal_output = self.reveal_head(shared_features)
        flag_output = self.flag_head(shared_features)
        return reveal_output, flag_output