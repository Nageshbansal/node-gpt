from torch import nn
from config import DROPOUT


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)
