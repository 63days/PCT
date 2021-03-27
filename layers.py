import torch
import torch.nn as nn
import torch.nn.functional as F


class LBR(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LBR, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x) #[B, N, C]
        if len(x.shape) == 3:
            x = x.transpose(1, 2) #[B, C, N]

        x = F.relu(self.bn(x), True)
        if len(x.shape) == 3:
            x = x.transpose(1, 2) #[B, N, C]
        return x


class LBRD(LBR):

    def __init__(self, in_channels, out_channels, p):
        super(LBRD, self).__init__(in_channels, out_channels)
        self.dp = nn.Dropout(p)

    def forward(self, x):
        x = super().forward(x)
        x = self.dp(x)
        return x


class NaiveInputEmbedding(nn.Module):

    def __init__(self, d_i, d_e=128):
        super(NaiveInputEmbedding, self).__init__()
        self.lbr1 = LBR(in_channels=d_i, out_channels=d_e)
        self.lbr2 = LBR(in_channels=d_e, out_channels=d_e)

    def forward(self, x):
        x = self.lbr1(x)
        x = self.lbr2(x)
        return x


class SA(nn.Module):

    def __init__(self, d_e, d_a):
        # input, v = d_e,     q, k = d_a
        # d_a = d_e / 4 for computational efficiency
        super(SA, self).__init__()
        self.d_e = d_e
        self.d_a = d_a

        self.linear_q = nn.Linear(d_e, d_a)
        self.linear_k = nn.Linear(d_e, d_a)
        self.linear_v = nn.Linear(d_e, d_e)
        self.lbr = LBR(in_channels=d_e, out_channels=d_e)

    def forward(self, x):
        q = self.linear_q(x)  # [B, N, da]
        k = self.linear_k(x)  # [B, N, da]
        v = self.linear_v(x)  # [B, N, de]
        score_tilde = torch.bmm(q, k.transpose(1, 2))
        score_bar = score_tilde / self.d_a
        score = torch.softmax(score_bar, dim=2) #[B, N, N]
        attn = torch.bmm(score, v) #[B, N, de]
        out = self.lbr(attn) + x

        return out


class OA(SA): # Offset-Attention

    def __init__(self, d_e, d_a):
        super(OA, self).__init__(d_e=d_e, d_a=d_a)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        score_tilde = torch.bmm(q, k.transpose(1, 2))
        score_bar = torch.softmax(score_tilde, dim=1)
        score = score_bar / torch.sum(score_bar, 2, keepdim=True)
        attn = torch.bmm(score, v)
        out = self.lbr(x - attn) + x

        return out

