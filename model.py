import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    def __init__(self, 
                 in_size, 
                 out_size, 
                 etypes, 
                 bias=True,
                 self_loop=True, 
                 dropout=0.0):
        
        super(HeteroRGCNLayer, self).__init__()
        self.bias = bias
        self.self_loop = self_loop
        
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)
        
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_size, out_size))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        
        hs = {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}
        for ntype in hs:
            h = hs[ntype]
            # apply bias and activation
            if self.self_loop:
                h = h + torch.matmul(feat_dict[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            h = self.dropout(h)
            hs[ntype] = h
        return hs
    
class HeteroRGCN(nn.Module):
    def __init__(self, 
                G, 
                in_size, 
                hidden_size, 
                out_size,
                bias=True,
                self_loop=True, 
                dropout=0.0):

        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes, bias, self_loop, dropout)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes, bias, self_loop, dropout)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict
