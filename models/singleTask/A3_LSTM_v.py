import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import TextSubNet


class A3_LSTM_v(nn.Module):
    """Video-only reproduction model."""

    def __init__(self, args):
        super().__init__()
        _, _, self.video_in = args.feature_dims
        _, _, self.video_hidden = args.hidden_dims
        self.video_out = args.video_out
        self.post_fusion_dim = args.post_fusion_dim
        _, self.video_prob, _, self.post_fusion_prob = args.dropouts

        self.video_subnet = TextSubNet(self.video_in, self.video_hidden, self.video_out, dropout=self.video_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.video_out, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        video_h = self.video_subnet(video_x)
        x = self.post_fusion_dropout(video_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        return {'M': self.post_fusion_layer_3(x)}
