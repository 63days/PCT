import torch
import torch.nn as nn
from layers import *


class NPCT(nn.Module):

    def __init__(self, d_i, d_e, d_a, n_classes, p):
        super(NPCT, self).__init__()
        self.input_embedding = NaiveInputEmbedding(d_i=d_i, d_e=d_e)
        self.sa1 = SA(d_e=d_e, d_a=d_a)
        self.sa2 = SA(d_e=d_e, d_a=d_a)
        self.sa3 = SA(d_e=d_e, d_a=d_a)
        self.sa4 = SA(d_e=d_e, d_a=d_a)
        self.linear = nn.Linear(d_e * 4, 1024)
        self.lbrd1 = LBRD(2048, 256, p)
        self.lbrd2 = LBRD(256, 256, p)
        self.final_linear = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.input_embedding(x)
        attn1 = self.sa1(x)
        attn2 = self.sa2(attn1)
        attn3 = self.sa3(attn2)
        attn4 = self.sa4(attn3)
        attns = torch.cat([attn1, attn2, attn3, attn4], dim=-1)
        linear_output = self.linear(attns) #[B, N, 1024]
        max_pool = linear_output.max(1)[0]
        average_pool = linear_output.mean(1)
        global_feature = torch.cat([max_pool, average_pool], dim=-1) #[B, 2048]
        x = self.lbrd1(global_feature)
        x = self.lbrd2(x)
        x = self.final_linear(x)
        return x


class SPCT(nn.Module):

    def __init__(self, d_i, d_e, d_a, n_classes, p):
        super(SPCT, self).__init__()
        self.input_embedding = NaiveInputEmbedding(d_i, d_e)
        self.oa1 = OA(d_e, d_a)
        self.oa2 = OA(d_e, d_a)
        self.oa3 = OA(d_e, d_a)
        self.oa4 = OA(d_e, d_a)
        self.linear = nn.Linear(d_e * 4, 1024)
        self.lbrd1 = LBRD(2048, 256, p)
        self.lbrd2 = LBRD(256, 256, p)
        self.final_linear = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.input_embedding(x)
        attn1 = self.oa1(x)
        attn2 = self.oa2(attn1)
        attn3 = self.oa3(attn2)
        attn4 = self.oa4(attn3)
        attns = torch.cat([attn1, attn2, attn3, attn4], dim=-1)
        linear_output = self.linear(attns)  # [B, N, 1024]
        max_pool = linear_output.max(1)[0]
        average_pool = linear_output.mean(1)
        global_feature = torch.cat([max_pool, average_pool], dim=-1)  # [B, 2048]
        x = self.lbrd1(global_feature)
        x = self.lbrd2(x)
        x = self.final_linear(x)
        return x