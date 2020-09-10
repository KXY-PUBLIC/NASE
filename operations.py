import torch
import torch.nn as nn
from torch.nn import functional as F

OPS = {
    'none': lambda emb_s, n, config: Zero(n),
    'skip_connect': lambda emb_s, n, config: Identity(n),
    'sep_conv_3x3_2d': lambda emb_s, n, config: SepConv2d(emb_s, n, 3, 1),
    'sep_conv_5x5_2d': lambda emb_s, n, config: SepConv2d(emb_s, n, 5, 2),
    'dil_conv_3x3_2d': lambda emb_s, n, config: DilConv2d(emb_s, n, 3, 2, 2),
    'sep_conv_3x3_1d': lambda emb_s, n, config: SepConv1d(emb_s, n, 2, 0),
    'TransE': lambda emb_s, n, config: TransE(emb_s, n),
    'TransR': lambda emb_s, n, config: TransR(emb_s, n, config),
    'TransH': lambda emb_s, n, config: TransH(emb_s, n, config),

    'Trans_cal': lambda emb_s, n, config: Trans_cal(emb_s, n, config),
    'Conv_cal': lambda emb_s, n, config: Conv_cal(emb_s, n, config),
    'Fc_cal': lambda emb_s, n, config: Fc_cal(emb_s, n, config),
    'DistMult_cal': lambda emb_s, n, config: DistMult_cal(emb_s, n, config),
    'SimplE_cal': lambda emb_s, n, config: SimplE_cal(emb_s, n, config),
}


class Zero(nn.Module):

  def __init__(self, n):
    super(Zero, self).__init__()
    self.n = n

  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = x0
    elif self.n == 1:
      x = x1
    else:
      x = x2
    return x.mul(0.)


class Identity(nn.Module):

  def __init__(self, n):
    super(Identity, self).__init__()
    self.n = n

  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = x0
    elif self.n == 1:
      x = x1
    else:
      x = x2
    return x


class SepConv2d(nn.Module):

  def __init__(self, emb_s, n, kernel_size, padding):
    super(SepConv2d, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)


  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = torch.cat([x1, x2], 1)  # (b_s, 2, emb_s)
    elif self.n == 1:
      x = torch.cat([x0, x2], 1)
    else:
      x = torch.cat([x0, x1], 1)

    x = x.view(-1, 2, int(self.emb_s / 10), 10)  # (b_s, 2, emb_s/10, 10)
    x = self.conv(x)
    x = x.view(-1, 1, self.emb_s)

    return x


class DilConv2d(nn.Module):

  def __init__(self, emb_s, n, kernel_size, padding=2, dilation=2):
    super(DilConv2d, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)

  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = torch.cat([x1, x2], 1)  # (b_s, 2, emb_s)
    elif self.n == 1:
      x = torch.cat([x0, x2], 1)
    else:
      x = torch.cat([x0, x1], 1)
    x = x.view(-1, 2, int(self.emb_s / 10), 10)  # (b_s, 2, emb_s/10, 10)
    x = self.conv(x)
    x = x.view(-1, 1, self.emb_s)

    return x


class SepConv1d(nn.Module):

  def __init__(self, emb_s, n, kernel_size, padding):
    super(SepConv1d, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.conv = nn.Conv1d(emb_s, emb_s, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = torch.cat([x1, x2], 1).transpose(2, 1)  # (b_s, emb_s, 2)
    elif self.n == 1:
      x = torch.cat([x0, x2], 1).transpose(2, 1)
    else:
      x = torch.cat([x0, x1], 1).transpose(2, 1)

    x = self.conv(x)  # (b_s, emb_s, 1)
    x = x.view(-1, 1, self.emb_s)

    return x


class TransE(nn.Module):

  def __init__(self, emb_s, n):
    super(TransE, self).__init__()
    self.n = n
    self.emb_s = emb_s

  def forward(self, x0, x1, x2, rel):
    if self.n == 0:
      x = x2 - x1
    elif self.n == 1:
      x = x2 - x0
    else:
      x = x0 + x1

    return x


class TransR(nn.Module):

  def __init__(self, emb_s, n, config=None):
    super(TransR, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.rel_cnt = config.rel_cnt
    self.transfer_matrix = nn.Embedding(self.rel_cnt, self.emb_s * self.emb_s)

  def forward(self, x0, x1, x2, rel):
    r_transfer = self.transfer_matrix(rel)
    r_transfer = r_transfer.view(-1, self.emb_s, self.emb_s)
    x0 = torch.matmul(x0, r_transfer)
    x2 = torch.matmul(x2, r_transfer)
    if self.n == 0:
      x = x2 - x1
      x = torch.matmul(x, torch.inverse(r_transfer))
    elif self.n == 1:
      x = x2 - x0
    else:
      x = x0 + x1
      x = torch.matmul(x, torch.inverse(r_transfer))

    return x


class TransH(nn.Module):

  def __init__(self, emb_s, n, config=None):
    super(TransH, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.rel_cnt = config.rel_cnt
    self.norm_vector = nn.Embedding(self.rel_cnt, self.emb_s)

  def forward(self, x0, x1, x2, rel):
    r_norm = self.norm_vector(rel).unsqueeze(1)
    x0 = x0 - torch.sum(x0 * r_norm, -1, True) * r_norm
    x2 = x2 - torch.sum(x2 * r_norm, -1, True) * r_norm
    if self.n == 0:
      x = x2 - x1
    elif self.n == 1:
      x = x2 - x0
    else:
      x = x0 + x1

    return x


class Trans_cal(nn.Module):

  def __init__(self, emb_s, n, config):
    super(Trans_cal, self).__init__()
    self.n = n
    self.emb_s = emb_s

  def forward(self, x0, x1, x2, rel):
    x = x0 + x1 - x2
    x = x.squeeze(1)
    score = torch.norm(x, p=1, dim=1).unsqueeze(1)

    return score


class Conv_cal(nn.Module):

    def __init__(self, emb_s, n, config):
        super(Conv_cal, self).__init__()
        self.n = n
        self.emb_s = emb_s
        self.conv_layer = nn.Conv2d(1, config.out_channels, (3, 3),
                                    padding=(1, 0))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(config.dropout)
        self.non_linearity = nn.ReLU()

        self.classifier = nn.Linear(self.emb_s * config.out_channels, 1)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.414)

    def forward(self, x0, x1, x2, rel):
        x = torch.cat([x0, x1, x2], 1)  # [b_s, 3, emb_s]
        # print("stacked_inputs", stacked_inputs.size())

        conv_input = x.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)  # [b_s, 1, emb_s, 3]
        x = self.dropout(self.non_linearity(self.conv_layer(conv_input)))  # [b_s, channel, emb_s, 1]

        x = x.view(x.size(0), -1)  # [b_s, emb_s * channel]
        score = self.classifier(x)
        #print("Conv_cal", score.size())

        return score


class Fc_cal(nn.Module):

    def __init__(self, emb_s, n, config):
        super(Fc_cal, self).__init__()
        self.n = n
        self.emb_s = emb_s

        self.classifier = nn.Linear(self.emb_s * 3, 1)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.414)

    def forward(self, x0, x1, x2, rel):
        x = torch.cat([x0, x1, x2], 1)  # [b_s, 3, emb_s]
        x = x.view(x.size(0), -1)  # [b_s, emb_s * 3]
        score = self.classifier(x)
        #print("Fc_cal", score.size())

        return score

class DistMult_cal(nn.Module):

  def __init__(self, emb_s, n, config):
    super(DistMult_cal, self).__init__()
    self.n = n
    self.emb_s = emb_s

  def forward(self, x0, x1, x2, rel):
    x = x0 * x1 * x2
    x = x.squeeze(1)
    score = x.sum(dim=1).unsqueeze(-1)
    #score = F.softmax(score, dim=0)
    #print("DistMult_cal", score.size())

    return score

class SimplE_cal(nn.Module):

  def __init__(self, emb_s, n, config):
    super(SimplE_cal, self).__init__()
    self.n = n
    self.emb_s = emb_s
    self.rel_cnt = config.rel_cnt
    self.rel_inv_embeddings = nn.Embedding(self.rel_cnt, self.emb_s)

  def forward(self, x0, x1, x2, rel):
    r_inv = self.rel_inv_embeddings(rel)
    h, r, t = x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)
    score = (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2
    score = score.unsqueeze(-1)
    #score = F.softmax(score, dim=0)
    #print("SimplE_cal", score.size())

    return score