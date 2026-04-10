import torch
import torch.nn as nn
import torch.nn.functional as F

from ..subNets import TextSubNet


class A4_LF_LSTM_la(nn.Module):
    """Text-audio reproduction model."""

    def __init__(self, args):
        super().__init__()
        self.text_in, self.audio_in, _ = args.feature_dims
        self.text_hidden, self.audio_hidden, _ = args.hidden_dims
        self.text_out = args.text_out
        self.audio_out = args.audio_out
        self.post_fusion_dim = args.post_fusion_dim
        self.audio_prob, _, self.text_prob, self.post_fusion_prob = args.dropouts

        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.audio_subnet = TextSubNet(self.audio_in, self.audio_hidden, self.audio_out, dropout=self.audio_prob)
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.audio_out, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        audio_h = self.audio_subnet(audio_x)
        text_h = self.text_subnet(text_x)
        fusion_h = torch.cat([audio_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        return {'M': self.post_fusion_layer_3(x)}
