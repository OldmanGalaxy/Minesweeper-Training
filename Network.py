import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self, input_size: int):
        super(Network, self).__init__()
        self.rows = int(input_size ** 0.5)
        self.cols = self.rows
        
        self.convolution = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )
        
        conv_output_size = 128 * self.rows * self.cols
        
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.ReLU()
        )
        
        self.positional_encoding = nn.Parameter(torch.randn(1, 128, self.rows, self.cols) * 0.01)
        
        self.attention = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        
        self.reveal_head = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        
        self.flag_head = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0) if len(x.shape) > 3 else 1
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        x = self.convolution(x)
        x = x + self.positional_encoding.expand(batch_size, -1, -1, -1)
        x = x.view(batch_size, -1)
        
        shared_features = self.shared_layers(x)
        attention_weights = self.attention(shared_features)
        weighted_features = shared_features * attention_weights
        
        reveal_output = self.reveal_head(weighted_features)
        flag_output = self.flag_head(weighted_features)
        
        if batch_size == 1:
            reveal_output = reveal_output.squeeze(0)
            flag_output = flag_output.squeeze(0)
        return reveal_output, flag_output