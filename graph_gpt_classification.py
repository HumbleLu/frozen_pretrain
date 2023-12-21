import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class Graph_GPT_Classification(nn.Module):
    def __init__(self, gpt_model, num_features, hidden_dim_1, num_classes, frozen = True):
        super().__init__()
        self.n_embed = gpt_model.config.n_embd
        self.num_classes = num_classes
        
        self.g_conv_1 = GCNConv(num_features, self.n_embed)

        self.transformer_layers = gpt_model.h
        # freezing transformer layer
        if frozen:
            for params in self.transformer_layers.parameters():
                params.requires_grad = False

        self.g_conv_2 = GCNConv(self.n_embed, num_classes)
        #self.ln = nn.Linear(self.n_embed, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        n_nodes = x.shape[0]
        
        x = self.g_conv_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = x.view([n_nodes, 1, -1])

        for i in range(len(self.transformer_layers)):
            x = self.transformer_layers[i](x)[0]

        x = x.view([n_nodes, -1])
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.g_conv_2(x, edge_index)
        
        x = F.log_softmax(x, dim = 1)
        
        return(x)