

import torch.nn as nn
from models.subNets import AlignSubNet
from models.singleTask.A1_LSTM_l import A1_LSTM_l
from models.singleTask.A2_LSTM_a import A2_LSTM_a
from models.singleTask.A3_LSTM_v import A3_LSTM_v
from models.singleTask.A4_LF_LSTM_la import A4_LF_LSTM_la
from models.singleTask.A5_LF_LSTM_lv import A5_LF_LSTM_lv
from models.singleTask.A6_LF_LSTM_av import A6_LF_LSTM_av
from models.singleTask.A7_LF_LSTM_lav import A7_LF_LSTM_lav
from models.singleTask.A8_LF_MLP_lav import A8_LF_MLP_lav
from models.singleTask.A9_EF_MLP_lav import A9_EF_MLP_lav
from models.singleTask.A10_EF_LSTM_lav import A10_EF_LSTM_lav
from models.singleTask.A11_MFN_lav import A11_MFN_lav


class AMIO(nn.Module):

    MODEL_MAP = {
        # ===================== Single-modality models =====================
        'A1_LSTM_l': A1_LSTM_l,  # Text-only LSTM (language features)
        'A2_LSTM_a': A2_LSTM_a,  # Audio-only LSTM (acoustic features)
        'A3_LSTM_v': A3_LSTM_v,  # Visual-only LSTM (visual features)
        # ===================== Dual-modality models =====================
        'A4_LF_LSTM_la': A4_LF_LSTM_la,  # Text + Audio (late fusion)
        'A5_LF_LSTM_lv': A5_LF_LSTM_lv,  # Text + Visual (late fusion)
        'A6_LF_LSTM_av': A6_LF_LSTM_av,  # Audio + Visual (late fusion)
        # ===================== Multimodal (tri-modal) models =====================
        'A7_LF_LSTM_lav': A7_LF_LSTM_lav,
        'A8_LF_MLP_lav': A8_LF_MLP_lav,
        'A9_EF_MLP_lav': A9_EF_MLP_lav,
        'A10_EF_LSTM_lav': A10_EF_LSTM_lav,
        'A11_MFN_lav': A11_MFN_lav,
        }

    def __init__(self, args):
        """Initialize the selected multimodal model with optional alignment."""
        super(AMIO, self).__init__()

        # Flag for whether input features need alignment
        self.need_model_aligned = args['need_model_aligned']

        # Initialize alignment subnetwork if needed
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')  # Using average pooling alignment

            # Update sequence lengths if specified in args
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()

        # Instantiate the selected model
        lastModel = self.MODEL_MAP[args['model_name']]  # Get model class from registry
        self.model_name = args['model_name']  # Store model name for reference
        self.Model = lastModel(args)  # Initialize the actual model

    def forward(self, text_x, audio_x, video_x, *args,**kwargs):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x,  *args, **kwargs)
