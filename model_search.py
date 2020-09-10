import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype, CLASSIFIER


class Architect(object):

    def __init__(self, model, args):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.weight_decay)

    def step(self, batch_triples_valid, batch_labels_valid):
        self.optimizer.zero_grad()
        loss = self.model._loss(batch_triples_valid, batch_labels_valid)
        loss.backward()
        self.optimizer.step()


class MixedOp(nn.Module):

    def __init__(self, emb_s, n, config=None, do_cal=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.do_cal = do_cal
        if do_cal:
            all_ops = CLASSIFIER
        else:
            all_ops = PRIMITIVES
        for primitive in all_ops:
            op = OPS[primitive](emb_s, n, config)
            self._ops.append(op)

    def forward(self, x0, x1, x2, rel, weights):
        res = []
        i = 0
        for w, op in zip(weights, self._ops):
            tmp = w * op(x0, x1, x2, rel)
            #if self.do_cal:
                #print("tmp", i, w, tmp)
            res.append(tmp)
            i += 1

        return sum(res)


class Cell(nn.Module):

    def __init__(self, emb_s, steps, config=None):
        super(Cell, self).__init__()
        self._steps = steps
        self.config = config

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            op = MixedOp(emb_s, i, self.config)
            self._ops.append(op)

        self.gate = nn.Linear(emb_s * 2, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.gate.weight, gain=1.414)

    def forward(self, s0, s1, s2, rel, weights):
        s_list = [s0, s1, s2]
        for i in range(self._steps):
            op_s = self._ops[i](s0, s1, s2, rel, weights[i])
            #print("op", op_s.size(), s_list[i].size())
            tmp = self.gate(torch.cat([op_s, s_list[i]], -1).squeeze(1))
            weight = self.sigmoid(tmp).unsqueeze(1)
            s_list[i] = weight * op_s + (1 - weight) * s_list[i]

        s0, s1, s2 = s_list

        return s0, s1, s2


class KG_search(nn.Module):

    def __init__(self, entity_emb, relation_emb, config=None):
        super(KG_search, self).__init__()
        self._layers = config.layers
        self._config = config
        self.emb_s = config.embedding_size
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self._steps = 3
        self.valid_invalid_ratio = config.valid_invalid_ratio
        self.do_margin_loss = config.do_margin_loss

        self._config.rel_cnt = entity_emb.shape[0]
        self._config.ent_cnt = entity_emb.shape[0]

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.cells = nn.ModuleList()
        for i in range(self._layers):
            cell = Cell(self.emb_s, self._steps, self._config)
            self.cells += [cell]

        self.dropout = nn.Dropout(config.dropout)
        self.cal_op = MixedOp(self.emb_s, 0, config=self._config, do_cal=True)

        if self.do_margin_loss:
            self.margin = config.margin
            self.loss = nn.MarginRankingLoss(margin=self.margin)
        else:
            self.loss = torch.nn.SoftMarginLoss()

        self._initialize_alphas()

    def forward(self, batch_inputs, batch_labels=None):
        #print("cal_alphas_normal", self.cal_alphas_normal)
        e1, rel, e2 = batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2]
        e1_emb = self.entity_embeddings[e1, :].unsqueeze(1)
        rel_emb = self.relation_embeddings[rel, :].unsqueeze(1)
        e2_emb = self.entity_embeddings[e2, :].unsqueeze(1)

        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas_normal, dim=-1)
            weights = weights[i * self._steps: (i + 1) * self._steps]
            e1_emb, rel_emb, e2_emb = cell(e1_emb, rel_emb, e2_emb, rel, weights)
            e1_emb = self.dropout(e1_emb)
            rel_emb = self.dropout(rel_emb)
            e2_emb = self.dropout(e2_emb)

        weights = F.softmax(self.cal_alphas_normal, dim=-1)
        output = self.cal_op(e1_emb, rel_emb, e2_emb, rel, weights[0])
        #print("output", output)

        if batch_labels is not None:
            if self.do_margin_loss:
                len_pos_triples = int(
                    batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))
                pos_norm = output[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)
                neg_norm = output[len_pos_triples:]
                y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
                loss = self.loss(pos_norm, neg_norm, y)
            else:
                loss = self.loss(output.view(-1), batch_labels.view(-1))
            return loss, 0

        if self.do_margin_loss:
            # y = -torch.ones(int(batch_inputs.size(0))).cuda()
            output = - output

        return output, 0

    def new(self):
        model_new = KG_search(self.entity_emb, self.relation_emb, config=self._config).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        for name, param in model_new.named_parameters():
            if param.requires_grad == False:
                param.requires_grad = True
        return model_new

    def _loss(self, batch_inputs, target):
        loss, _ = self(batch_inputs, target)
        return loss

    def _initialize_alphas(self):
        k = self._steps * self._layers
        num_ops = len(PRIMITIVES)
        cal_num_ops = len(CLASSIFIER)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.cal_alphas_normal = Variable(1e-3 * torch.randn(1, cal_num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal, self.cal_alphas_normal
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            all_step = weights.shape[0]
            if all_step == 1:
                all_ops = CLASSIFIER
            else:
                all_ops = PRIMITIVES
            gene = []
            for i in range(all_step):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if all_step == 1 or k != all_ops.index('none'):
                        if k_best is None or W[k] > W[k_best]:
                            k_best = k
                gene.append((all_ops[k_best], i % 3))

            return gene

        #print("self.alphas_normal", self.alphas_normal)
        #print("genotype, cal_alphas_normal", self.cal_alphas_normal)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_normal.extend(_parse(F.softmax(self.cal_alphas_normal, dim=-1).data.cpu().numpy()))

        genotype = Genotype(normal=gene_normal)

        return genotype

