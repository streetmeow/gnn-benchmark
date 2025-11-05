
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        layers = []
        layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden_dim, hidden_dim))
        layers.append(GCNConv(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.act(x)
                x = self.drop(x)
        return x

def build_from_cfg(cfg, data):
    return GCN(cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim, cfg.model.num_layers, cfg.model.dropout)
