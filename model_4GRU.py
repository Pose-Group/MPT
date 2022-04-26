import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

class Graph():
    def __init__(self):
        self.get_edge()
    def get_edge(self):
        self.node_num = 25
        self_link = [(i, i) for i in range(self.node_num)]
        bone_link = [(0, 1), (1, 2), (2, 3), (3,4), (4,5), (0,6), (6,7), (7,8), (8,9), (9,10), (0,11), (11,12), (12,13), (13,14),
                     (12,15), (15,16), (16,17),(17,19), (17,18), (12,20), (20,21), (21,22), (22,23), (22,24)]
        self.edge = self_link + bone_link
        
        self.pre_sem_edge = [(2,7),(3,8),(16,21),(17,22)]
        A_ske = torch.zeros((self.node_num, self.node_num))
        for i, j in self.edge:
            A_ske[j, i] = 1
            A_ske[i, j] = 1
        self.A_ske = A_ske
        A_pre_sem = torch.zeros((self.node_num, self.node_num))
        for p, q in self.pre_sem_edge:
            A_pre_sem[p, q] = 1
            A_pre_sem[q, p] = 1
        self.A_pre_sem = A_pre_sem
        
        return A_ske, A_pre_sem

class SemskeConv(nn.Module):
    
    def __init__(self, in_features, out_features, node_num, bias=True):
        super(SemskeConv, self).__init__()
        self.node_num = node_num

        self.graph = Graph()
        
        A_ske = torch.tensor(self.graph.A_ske, dtype=torch.float, requires_grad=False) #As
        A_pre_sem = torch.tensor(self.graph.A_pre_sem, dtype=torch.float, requires_grad=False) #Ap
        A_sem = nn.Parameter(torch.zeros(node_num, node_num)) #Af
        adj = A_ske + A_pre_sem 
        self.adj = adj
        self.A_sem = A_sem
        self.M = nn.Parameter(torch.zeros(node_num, node_num))
        self.W = nn.Parameter(torch.zeros(node_num, node_num))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(node_num))
            stdv = 1. / math.sqrt(self.M.size(1))
            self.bias.data.uniform_(-stdv, stdv)
            
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        self.adj = torch.where(torch.isnan(self.adj), torch.full_like(self.adj, 0), self.adj).cuda()
        self.A_sem = nn.Parameter(torch.where(torch.isnan(self.A_sem), torch.full_like(self.A_sem, 0), self.A_sem)).cuda()
        self.W = nn.Parameter(torch.where(torch.isnan(self.W), torch.full_like(self.W, 0), self.W))
        self.M = nn.Parameter(torch.where(torch.isnan(self.M), torch.full_like(self.M, 0), self.M))
        Adj = self.adj + self.A_sem
        Adj = 0.50*(self.adj + self.adj.permute(1, 0))
        Adj_W = torch.mul(Adj, self.W)
        support = torch.matmul(input, Adj_W)
        output = torch.matmul(support, self.M)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class _GraphConv(nn.Module):
    def __init__(self, in_features, hidden_feature, node_num, p_dropout= 0.005):
        super(_GraphConv, self).__init__()
        
        self.gconv1 = SemskeConv(in_features, hidden_feature, node_num)
        self.bn = nn.BatchNorm1d(node_num* hidden_feature)

        self.gconv2 = SemskeConv(hidden_feature, in_features, node_num)

        self.tanh = nn.Tanh()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        
        y = self.gconv1(x)
        b, f, n= y.shape

        y = self.tanh(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.gconv2(y)

        b, f, n= y.shape
        y = self.tanh(y)
        y = y + x

        return y


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, node_num, batch_size):
        super(Generator, self).__init__()


        self.hidden_prev = nn.Parameter(torch.zeros(4, batch_size, hidden_size))
        

        self.GRU = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=4, dropout=0.05, batch_first = True)
                          
        
        self.GCN = _GraphConv(1, 10, node_num)
        
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden_size):

        # GCN block
        x = x.permute(0, 2, 1)
        GCN_set = self.GCN(x)
        
        
        x = GCN_set.reshape(x.shape[0],x.shape[1],x.shape[2])
        x = x.permute(0, 2, 1)
        

        out, h = self.GRU(x, self.hidden_prev)
        out = out.reshape(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, h

