import torch
import torch.nn as nn

class LSTMBranch(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, embedding_dim=128):
        super(LSTMBranch, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)

        out, (hidden, cell) = self.lstm(x)

        final_state = out[:,-1,:]

        x = self.fc(final_state)

        x = nn.functional.normalize(x, p=2, dim=1)

        return x
