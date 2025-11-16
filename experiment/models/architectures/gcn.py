from torch_geometric.nn import GCNConv
from experiment.models import BaseNN


class GCN(BaseNN):
    def _build_layers(self):
        # input → hidden
        self.layers.append(GCNConv(self.in_dim, self.hidden_dim))

        # hidden → hidden
        for _ in range(self.num_layers - 2):
            self.layers.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # hidden → output
        self.layers.append(GCNConv(self.hidden_dim, self.out_dim))
