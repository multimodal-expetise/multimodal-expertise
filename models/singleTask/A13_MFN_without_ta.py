"""
MFN ablation: remove T<->A interaction, keep T<->V + A<->V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class A13_MFN_without_ta(nn.Module):
    def __init__(self, args):
        super(A13_MFN_without_ta, self).__init__()
        self.d_l, self.d_a, self.d_v = args.feature_dims
        self.dh_l, self.dh_a, self.dh_v = args.hidden_dims
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = args.memsize
        output_dim = 1
        gammaInShape = self.mem_dim + self.mem_dim
        final_out = total_h_dim + self.mem_dim

        h_att1 = args.NN1Config["shapes"]
        h_att2 = args.NN2Config["shapes"]
        h_gamma1 = args.gamma1Config["shapes"]
        h_gamma2 = args.gamma2Config["shapes"]
        h_out = args.outConfig["shapes"]
        att1_dropout = args.NN1Config["drop"]
        att2_dropout = args.NN2Config["drop"]
        gamma1_dropout = args.gamma1Config["drop"]
        gamma2_dropout = args.gamma2Config["drop"]
        out_dropout = args.outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        # TV pair (kept)
        self.att1_TV = nn.Sequential(
            nn.Linear(2 * (self.dh_l + self.dh_v), h_att1),
            nn.ReLU(), nn.Dropout(att1_dropout),
            nn.Linear(h_att1, 2 * (self.dh_l + self.dh_v)))
        self.att2_TV = nn.Sequential(
            nn.Linear(2 * (self.dh_l + self.dh_v), h_att2),
            nn.ReLU(), nn.Dropout(att2_dropout),
            nn.Linear(h_att2, self.mem_dim), nn.Tanh())

        # AV pair (kept)
        self.att1_AV = nn.Sequential(
            nn.Linear(2 * (self.dh_a + self.dh_v), h_att1),
            nn.ReLU(), nn.Dropout(att1_dropout),
            nn.Linear(h_att1, 2 * (self.dh_a + self.dh_v)))
        self.att2_AV = nn.Sequential(
            nn.Linear(2 * (self.dh_a + self.dh_v), h_att2),
            nn.ReLU(), nn.Dropout(att2_dropout),
            nn.Linear(h_att2, self.mem_dim), nn.Tanh())

        # TA pair is omitted

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)
        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, text_x, audio_x, video_x):
        text_x = text_x.permute(1, 0, 2)
        audio_x = audio_x.permute(1, 0, 2)
        video_x = video_x.permute(1, 0, 2)
        n = text_x.size(1)
        t = text_x.size(0)
        device = text_x.device

        self.h_l = torch.zeros(n, self.dh_l, device=device)
        self.h_a = torch.zeros(n, self.dh_a, device=device)
        self.h_v = torch.zeros(n, self.dh_v, device=device)
        self.c_l = torch.zeros(n, self.dh_l, device=device)
        self.c_a = torch.zeros(n, self.dh_a, device=device)
        self.c_v = torch.zeros(n, self.dh_v, device=device)
        self.mem = torch.zeros(n, self.mem_dim, device=device)

        all_h_ls, all_h_as, all_h_vs = [], [], []
        all_mems = []

        for i in range(t):
            prev_c_l, prev_c_a, prev_c_v = self.c_l, self.c_a, self.c_v

            new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))

            cHat_sum = torch.zeros(n, self.mem_dim, device=device)

            # TA pair is skipped

            # TV pair
            pair_cStar_TV = torch.cat([new_c_l, new_c_v, prev_c_l, prev_c_v], dim=1)
            att_TV = F.softmax(self.att1_TV(pair_cStar_TV), dim=1)
            cHat_sum = cHat_sum + self.att2_TV(att_TV * pair_cStar_TV)

            # AV pair
            pair_cStar_AV = torch.cat([new_c_a, new_c_v, prev_c_a, prev_c_v], dim=1)
            att_AV = F.softmax(self.att1_AV(pair_cStar_AV), dim=1)
            cHat_sum = cHat_sum + self.att2_AV(att_AV * pair_cStar_AV)

            # Delta memory
            both = torch.cat([cHat_sum, self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1 * self.mem + gamma2 * cHat_sum
            all_mems.append(self.mem)

            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v

            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)

        last_hs = torch.cat([all_h_ls[-1], all_h_as[-1], all_h_vs[-1], all_mems[-1]], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return {'M': output, 'L': last_hs}
