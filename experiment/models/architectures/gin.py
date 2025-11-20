import torch.nn as nn
from torch_geometric.nn import GINConv
from experiment.models.base_nn import BaseNN


class GIN(BaseNN):
    def _build_layers(self):
        """
        GINConv 는 내부에 nn.Sequential 형태의 MLP 를 넣어줘야 해서
        GCN/GAT/SAGE 와 달리 'MLP wrapper' 를 만들어서 사용한다.
        """

        # --- 1) input → hidden ---
        mlp_in = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.layers.append(GINConv(mlp_in))

        # --- 2) hidden → hidden ---
        for _ in range(self.num_layers - 2):
            mlp_hidden = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.layers.append(GINConv(mlp_hidden))

        # --- 3) hidden → output ---
        mlp_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.layers.append(GINConv(mlp_out))
