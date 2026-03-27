import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .lstm_branch import LSTMBranch

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        self.cnn_branch = CNNBranch(embedding_dim=embedding_dim)
        self.lstm_branch = LSTMBranch(embedding_dim=embedding_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, mel, pitch):
        
        cnn_features = self.cnn_branch(mel)

        lstm_features = self.lstm_branch(pitch)

        combined_features = torch.cat((cnn_features, lstm_features), dim=1)

        fused_embedding = self.fusion(combined_features)

        final_output = nn.functional.normalize(fused_embedding, p=2, dim=1)

        return final_output
