import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import TextSubNet


class A2_LSTM_a(nn.Module):
    """Audio-only reproduction model."""

    def __init__(self, args):
        super().__init__()
        _, self.audio_in, _ = args.feature_dims
        _, self.audio_hidden, _ = args.hidden_dims
        self.audio_out = args.audio_out
        self.post_fusion_dim = args.post_fusion_dim
        self.audio_prob, _, _, self.post_fusion_prob = args.dropouts

        self.audio_subnet = TextSubNet(self.audio_in, self.audio_hidden, self.audio_out, dropout=self.audio_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.audio_out, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        audio_h = self.audio_subnet(audio_x)
        x = self.post_fusion_dropout(audio_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        return {'M': self.post_fusion_layer_3(x)}
