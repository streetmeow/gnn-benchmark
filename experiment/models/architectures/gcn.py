from torch_geometric.nn import GCNConv
from experiment.models.base_nn import BaseNN
from torch.nn import BatchNorm1d


class GCN(BaseNN):
    def _build_layers(self):
        # input → hidden
        self.layers.append(GCNConv(self.in_dim, self.hidden_dim))
        if self.use_batchnorm:
            self.bns.append(BatchNorm1d(self.hidden_dim))

        # hidden → hidden
        for _ in range(self.num_layers - 2):
            self.layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            if self.use_batchnorm:
                self.bns.append(BatchNorm1d(self.hidden_dim))

        # hidden → output
        self.layers.append(GCNConv(self.hidden_dim, self.out_dim))
