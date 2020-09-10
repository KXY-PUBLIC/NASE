import torch
import torch.nn as nn
from operations import *
from torch.nn import functional as F


class Cell(nn.Module):

  def __init__(self, op_names, indices, emb_s, steps, config=None):
    super(Cell, self).__init__()
    self._steps = steps
    self.emb_s = emb_s
    self._ops = nn.ModuleList()
    self.config = config
    for name, index in zip(op_names, indices):
      op = OPS[name](emb_s, index, self.config)
      self._ops += [op]

    self.gate = nn.Linear(self.emb_s * 2, 1)
    self.sigmoid = nn.Sigmoid()
    nn.init.xavier_uniform_(self.gate.weight, gain=1.414)

  def forward(self, s0, s1, s2, rel, idx):
    start = idx * self._steps
    s_list = [s0, s1, s2]
    for i in range(self._steps):
      op_s = self._ops[i + start](s0, s1, s2, rel)
      tmp = self.gate(torch.cat([op_s, s_list[i]], -1).squeeze(1))
      weight = self.sigmoid(tmp).unsqueeze(1)
      s_list[i] = weight * op_s + (1 - weight) * s_list[i]

    s0, s1, s2 = s_list

    return s0, s1, s2

class NASE(nn.Module):

  def __init__(self, entity_emb, relation_emb, genotype, config=None):
    super(NASE, self).__init__()
    self._layers = config.layers
    self._config = config
    self.emb_s = config.embedding_size
    self._steps = 3
    self.valid_invalid_ratio = config.valid_invalid_ratio
    self.do_margin_loss = config.do_margin_loss

    self._config.rel_cnt = entity_emb.shape[0]
    self._config.ent_cnt = entity_emb.shape[0]
    self.entity_embeddings = nn.Parameter(entity_emb)
    self.relation_embeddings = nn.Parameter(relation_emb)

    op_names, indices = zip(*genotype.normal)
    self.cell = Cell(op_names[:-1], indices[:-1], self.emb_s, self._steps, self._config)

    self.dropout = nn.Dropout(config.dropout)
    self.cal_op = OPS[op_names[-1]](self.emb_s, 0, self._config)

    if self.do_margin_loss:
        self.margin = config.margin
        self.loss = nn.MarginRankingLoss(margin=self.margin)
    else:
        self.loss = torch.nn.SoftMarginLoss()

  def forward(self, batch_inputs, batch_labels=None):
    e1, rel, e2 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
    e1_emb = self.entity_embeddings[e1, :].unsqueeze(1)
    rel_emb = self.relation_embeddings[rel, :].unsqueeze(1)
    e2_emb = self.entity_embeddings[e2, :].unsqueeze(1)

    for idx in range(self._layers):
      cell = self.cell
      e1_emb, rel_emb, e2_emb = cell(e1_emb, rel_emb, e2_emb, rel, idx)
      e1_emb = self.dropout(e1_emb)
      rel_emb = self.dropout(rel_emb)
      e2_emb = self.dropout(e2_emb)

    output = self.cal_op(e1_emb, rel_emb, e2_emb, rel)  # (b_s, 1)

    if batch_labels is not None:
      if self.do_margin_loss:
          len_pos_triples = int(batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))
          pos_norm = output[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)
          neg_norm = output[len_pos_triples:]
          y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
          loss = self.loss(pos_norm, neg_norm, y)
      else:
          loss = self.loss(output.view(-1), batch_labels.view(-1))
      return loss, 0

    if self.do_margin_loss:
        #y = -torch.ones(int(batch_inputs.size(0))).cuda()
        output = - output
    return output, 0

