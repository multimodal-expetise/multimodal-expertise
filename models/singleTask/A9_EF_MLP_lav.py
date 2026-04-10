import torch
import torch.nn as nn


class A9_EF_MLP_lav(nn.Module):
    """Early-fusion MLP reproduction model."""

    def __init__(self, args):
        super().__init__()
        text_in, audio_in, video_in = args.feature_dims
        in_size = text_in + audio_in + video_in
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout

        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_layers),
            nn.ReLU(),
            nn.Linear(num_layers, 1),
        )
        self.norm = nn.BatchNorm1d(in_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        x = torch.cat([text_x, audio_x, video_x], dim=-1)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return {'M': self.mlp(x)}
