import torch
import torch.nn as nn

__all__ = ['AlignSubNet']


class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len + 1, num_layers=2,
                                                        batch_first=True)
        self.out_seq_len = out_seq_len

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(
            pred_output_position_inclu_blank)
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :,
                                    1:]
        prob_pred_output_position = prob_pred_output_position.transpose(1, 2)
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x)


        return pseudo_aligned_out


class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d']

        in_dim_t, in_dim_a, in_dim_v = args.feature_dims
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.mode = mode
        print("mode: the way of aligning :", self.mode)


        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool

        }



    def __getattr__(self, name):

        if 'avg_pool' in name:
            return self.__avg_pool

        # 必须调用父类，否则 PyTorch 的原生功能会失效
        try:
            return super(AlignSubNet, self).__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ----------------------------

    def get_seq_len(self):
        return self.dst_len

    def __ctc(self, text_x, audio_x, video_x):

        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x

    def __avg_pool(self, text_x, audio_x, video_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len: return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            return x.mean(dim=1)

        return align(text_x), align(audio_x), align(video_x)

    def forward(self, text_x, audio_x, video_x):
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        if self.mode == 'avg_pool': return self.__avg_pool(text_x, audio_x, video_x)
        return self.ALIGN_WAY[self.mode](text_x, audio_x, video_x)