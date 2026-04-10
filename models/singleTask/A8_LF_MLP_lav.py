import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Simple feed-forward block used before fusion."""

    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.drop(x)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return F.relu(self.linear_3(x))


class A8_LF_MLP_lav(nn.Module):
    """Late fusion using MLP."""

    def __init__(self, args):
        super().__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.post_fusion_dim = args.post_fusion_dim
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        self.audio_subnet = MLP(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = MLP(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = MLP(self.text_in, self.text_hidden, self.text_prob)

        total_fusion_dim = self.audio_hidden + self.video_hidden + self.text_hidden
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(total_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        audio_h = self.audio_subnet(audio_x).mean(dim=1, keepdim=True)
        video_h = self.video_subnet(video_x).mean(dim=1, keepdim=True)
        text_h = self.text_subnet(text_x).mean(dim=1, keepdim=True)
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        return {'M': self.post_fusion_layer_3(x).squeeze(1)}
