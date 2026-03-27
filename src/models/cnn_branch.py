import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CNNBranch, self).__init__()

        self.conv_blocks = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):

        x = self.conv_blocks(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x,1)

        x = self.fc(x)

        x = nn.functional.normalize(x, p=2, dim=1)
    
        return x
