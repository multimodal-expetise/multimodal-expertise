import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import TextSubNet


class A1_LSTM_l(nn.Module):
    """Text-only reproduction model."""

    def __init__(self, args):
        super().__init__()
        self.text_in, _, _ = args.feature_dims
        self.text_hidden, _, _ = args.hidden_dims
        self.text_out = args.text_out
        self.post_fusion_dim = args.post_fusion_dim
        _, _, self.text_prob, self.post_fusion_prob = args.dropouts

        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        text_h = self.text_subnet(text_x)
        x = self.post_fusion_dropout(text_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        return {'M': self.post_fusion_layer_3(x)}
