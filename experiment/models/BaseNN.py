import torch.nn as nn
import torch.nn.functional as F


class BaseNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout=0.5,
        activation="relu"
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_name = activation.lower()

        # activation 함수 선택
        self.activation = self._get_activation(self.act_name)

        # 하위 모델이 conv layer 를 구성함
        self.layers = nn.ModuleList()
        self._build_layers()

    def _get_activation(self, name):
        if name == "relu":
            return F.relu
        elif name == "gelu":
            return F.gelu
        elif name == "elu":
            return F.elu
        raise ValueError(f"Unknown activation: {name}")

    def _build_layers(self):
        raise NotImplementedError

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:          # 마지막 레이어 제외
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
