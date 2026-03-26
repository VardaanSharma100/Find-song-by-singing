import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()

        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):

        loss = self.triplet_loss(anchor, positive, negative)

        return loss