
from torch import nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, num_layers=2, dropout=0.6, concat=True):
        super().__init__()
        layers = []
        din = in_dim
        for _ in range(num_layers - 1):
            layers.append(GATConv(din, hidden_dim, heads=heads, dropout=dropout, concat=concat))
            din = hidden_dim * (heads if concat else 1)
        layers.append(GATConv(din, out_dim, heads=1, concat=False, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ELU()
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
    return GAT(cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim, heads=cfg.model.heads, num_layers=cfg.model.num_layers, dropout=cfg.model.dropout, concat=cfg.model.concat)
