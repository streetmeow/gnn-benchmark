import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from experiment.models.base_nn import BaseNN


class SGC(BaseNN):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout=0.5,
        activation="relu",
        use_batchnorm=False,
        k_value=2
    ):
        self.k_value = k_value
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout, activation)

    def _build_layers(self):
        # SGC는 conv 하나만 필요함
        # K는 몇 번 propagation 할지 (보통 2~3)

        # input → output (SGC는 중간 hidden 없음)
        self.layers.append(SGConv(self.in_dim, self.out_dim, K=2))

    def forward(self, x, edge_index):
        x = self.layers[0](x, edge_index)
        return x
