import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size: int):
        super(Network, self).__init__()
        self.rows = int(input_size ** 0.5)
        self.cols = self.rows
        
        self.convolution = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        conv_output_size = 64 * self.rows * self.cols
        
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
        
        self.reveal_head = nn.Sequential(
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
        
        self.flag_head = nn.Sequential(
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        
        shared_features = self.shared_layers(x)
        attention_weights = self.attention(shared_features)
        
        reveal_output = self.reveal_head(shared_features) * attention_weights
        flag_output = self.flag_head(shared_features) * attention_weights
        
        if x.size(0) == 1:
            reveal_output = reveal_output.squeeze(0)
            flag_output = flag_output.squeeze(0)
        return reveal_output, flag_output