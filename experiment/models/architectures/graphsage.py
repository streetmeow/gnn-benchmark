from torch_geometric.nn import SAGEConv
from experiment.models.BaseNN import BaseNN


class GraphSAGE(BaseNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.5, activation="relu", aggr="mean"):
        self.aggr = aggr
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout, activation)

    def _build_layers(self):
        # input → hidden
        self.layers.append(
            SAGEConv(self.in_dim, self.hidden_dim, aggr=self.aggr)
        )

        # hidden → hidden
        for _ in range(self.num_layers - 2):
            self.layers.append(
                SAGEConv(self.hidden_dim, self.hidden_dim, aggr=self.aggr)
            )

        # hidden → output
        self.layers.append(
            SAGEConv(self.hidden_dim, self.out_dim, aggr=self.aggr)
        )
