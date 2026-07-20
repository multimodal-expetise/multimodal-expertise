
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP (nn.Module):
    '''
    The subnetwork that is a fully-connected Multilayer Perceptron (MLP).
    '''

    def __init__(self, in_size, hidden_size, dropout):
        super(MLP, self).__init__()

        # MLP layers setup
        self.norm = nn.BatchNorm1d(in_size)  # Normalization to standardize input
        self.drop = nn.Dropout(p=dropout)  # Dropout layer to prevent overfitting

        # Fully connected layers
        self.linear_1 = nn.Linear(in_size, hidden_size)  # First fully connected layer
        self.linear_2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.linear_3 = nn.Linear(hidden_size, hidden_size)  # Third fully connected layer

    def forward(self, x):

        dropped = self.drop(x)

        y_1 = F.relu(self.linear_1(dropped))  # First hidden layer with ReLU activation
        y_2 = F.relu(self.linear_2(y_1))  # Second hidden layer with ReLU activation
        y_3 = F.relu(self.linear_3(y_2))  # Third hidden layer with ReLU activation

        return y_3


class A8_LF_MLP_lav(nn.Module):
    """
    late fusion using MLP
    """

    def __init__(self, args):
        super(A8_LF_MLP_lav, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims

        self.text_out = args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        output_dim = 1

        # define the pre-fusion subnetworks
        self.audio_subnet = MLP(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = MLP(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = MLP(self.text_in, self.text_hidden, self.text_prob)

        # Calculate total feature dimension after fusion
        total_fusion_dim = self.audio_hidden + self.video_hidden + self.text_hidden

        # define the post-fusion layers with the correct input dimension
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(total_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, output_dim)

    def forward(self, text_x, audio_x, video_x, control_x=None):
        # Pass through the subnetworks
        audio_h = self.audio_subnet(audio_x)  # [B, S, D]
        video_h = self.video_subnet(video_x)  # [B, S, D]
        text_h = self.text_subnet(text_x)     # [B, S, D]

        # Apply global pooling to reduce the sequence dimension (S) to 1
        audio_h = audio_h.mean(dim=1, keepdim=True)  # [B, 1, D]
        video_h = video_h.mean(dim=1, keepdim=True)  # [B, 1, D]
        text_h = text_h.mean(dim=1, keepdim=True)    # [B, 1, D]

        # Concatenate the outputs along the feature dimension (dim=-1)
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)  # [B, 1, D1 + D2 + D3]

        # Apply post-fusion layers
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)
        output = output.squeeze(1)  # Squeeze the second dimension

        res = {
            'M': output
        }
        return res

