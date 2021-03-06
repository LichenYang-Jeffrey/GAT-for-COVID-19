import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from graph import readout


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, nlmphid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(nhid, nout, dropout=dropout, alpha=alpha, concat=False)

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(nout, nlmphid),
            torch.nn.ReLU(),
            torch.nn.Linear(nlmphid, nclass),
        )

    def forward(self, x, x_mask, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        att = self.attentions[0]
        x = torch.cat([att(x_, adj_).unsqueeze(0) for att in self.attentions for (x_, adj_) in zip(x,adj)], dim=0)
        # x = torch.bmm(adj, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([F.elu(self.out_att(x_, adj_)).unsqueeze(0) for (x_, adj_) in zip(x, adj)],dim=0)
        # return F.log_softmax(x, dim=1)
        x = readout(x, x_mask)
        x = self.MLP(x)
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

