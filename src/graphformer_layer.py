#Adding in the layers and modifying them to take in node features. This module is meant to work for node classification and is meant to work for 1 graph.

import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, 
          num_heads, 
          feat_dim, 
          num_in_degree, 
          hidden_dim, 
          n_layers, 
          graph_type='undirected', 
          num_out_degree=None,
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.feat_dim = feat_dim
        self.num_atoms = self.feat_dim[0] 
        self.hidden_dim = hidden_dim
        self.num_in_degree = num_in_degree
        self.graph_type = graph_type
        self.num_out_degree = num_out_degree
              

        # 1 for graph token
        self.atom_encoder = nn.Embedding(self.num_atoms + 1, self.hidden_dim, padding_idx=0)
        self.node_encoder = nn.Linear(self.feat_dim, self.hidden_dim)
        self.in_degree_encoder = nn.Embedding(self.num_in_degree, self.hidden_dim, padding_idx=0)
        if self.graph_type == 'directed':
          assert self.num_out_degree is not None
          self.out_degree_encoder = nn.Embedding(
            self.num_out_degree, self.hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature
