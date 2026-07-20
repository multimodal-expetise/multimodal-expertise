import torch
import torch.nn as nn
import torch.nn.functional as F




class A9_EF_MLP_lav (nn.Module):
    """
    Early fusion using LSTM and MLP for multimodal input.
    """

    def __init__(self, args):
        super(A9_EF_MLP_lav, self).__init__()
        text_in, audio_in, video_in = args.feature_dims

        in_size = text_in + audio_in + video_in

        input_len = in_size  # Total input length after fusion
        hidden_size = args.hidden_dims
        num_layers = args.num_layers
        dropout = args.dropout
        output_dim = 1  # For binary sentiment classification

        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_layers),
            nn.ReLU(),
            nn.Linear(num_layers, output_dim)
        )

        # BatchNorm and Dropout layers
        self.norm = nn.BatchNorm1d(input_len)  # Ensure this matches the input dimension
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Final output layer
        self.out = nn.Linear(hidden_size, output_dim)

        # Global Average Pooling Layer
        self.pool = nn.AdaptiveAvgPool1d(1)  # Adaptive Pooling to get the final prediction

    def forward(self, text_x, audio_x, video_x, control_x=None):
        # Early fusion of modalities

        x = torch.cat([text_x, audio_x, video_x], dim=-1)

        x = x.mean(dim=1)
        # Apply normalization
        x = self.norm(x)

        # Pass through MLP
        output = self.mlp(x)

        # Final output layer to produce the prediction
        res = {
            'M': output
        }
        return res