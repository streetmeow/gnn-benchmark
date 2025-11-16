from torch_geometric.nn import GATConv
from experiment.models import BaseNN


class GAT(BaseNN):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout=0.5,
        activation="relu",
        heads=4,
        concat=False
    ):
        self.heads = heads
        self.concat = concat
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout, activation)

    def _build_layers(self):
        # input → hidden
        self.layers.append(
            GATConv(self.in_dim, self.hidden_dim, heads=self.heads, concat=self.concat)
        )

        # hidden → hidden
        for _ in range(self.num_layers - 2):
            self.layers.append(
                GATConv(self.hidden_dim, self.hidden_dim, heads=self.heads, concat=self.concat)
            )

        # hidden → output (heads=1, concat=False)
        self.layers.append(
            GATConv(self.hidden_dim, self.out_dim, heads=1, concat=False)
        )
