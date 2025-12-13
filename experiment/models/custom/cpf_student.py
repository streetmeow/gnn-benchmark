# experiment/models/cpf_student.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class FeatureMLP(nn.Module):
    """
    CPF에서 FT(Feature-based Teacher) 역할을 하는 간단한 MLP.
    x -> logits_ft (N, C)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(x)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        out = self.fc2(h)
        return out


class PLPConv(nn.Module):
    """
    CPF 논문식 Label Propagation Layer:

        H^{k+1} = \hat{A} H^{k},
        \hat{A} = D^{-1/2} (A + I) D^{-1/2}

    - 입력: label_mat (N, C)  / edge_index (2, E)
    - 출력: propagated label_mat (N, C)
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        label_mat: torch.Tensor,      # (N, C)
        edge_index: torch.Tensor,     # (2, E)
        num_nodes: int,
    ) -> torch.Tensor:
        device = label_mat.device
        row, col = edge_index  # row: src, col: dst 라고 생각

        # self-loop 추가: A + I
        self_loop = torch.arange(num_nodes, device=device)
        row = torch.cat([row, self_loop], dim=0)
        col = torch.cat([col, self_loop], dim=0)

        # degree 계산: D
        one = torch.ones_like(row, dtype=label_mat.dtype)
        deg = scatter_add(one, row, dim=0, dim_size=num_nodes)  # (N,)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # \hat{A}의 scalar weight
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]    # (E,)

        # message passing: H^{k+1} = \hat{A} H^k
        out = torch.zeros_like(label_mat)               # (N, C)
        out.index_add_(
            0,
            col,
            label_mat[row] * norm.unsqueeze(-1),        # (E, C)
        )
        return out


class CPFStudent(nn.Module):
    """
    PyG-independent CPF Student 모델 (베이스라인).

    - FT branch: FeatureMLP(x) -> logits_ft
    - PLP branch: label_init를 \hat{A} 기반으로 plp_steps 만큼 propagation
                  (각 step마다 train node는 hard one-hot으로 고정)
    - Node-wise gate α: (N, 1) learnable parameter
        logits = σ(α) * plp_logits + (1 - σ(α)) * logits_ft

    주의:
    - forward 시그니처는 framework 규약에 맞게 (x, edge_index)만 받음.
    - label_init / train_mask / hard_one_hot 은 set_label_info()로
      trainer가 사전에 주입해준다.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_nodes: int,
        plp_steps: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.plp_steps = plp_steps

        # FT branch
        self.ft_mlp = FeatureMLP(in_dim, hidden_dim, out_dim, dropout=dropout)

        # PLP branch (논문식 propagation)
        self.plp_conv = PLPConv()

        # Node-wise gate α (N, 1)
        self.alpha = nn.Parameter(torch.zeros(num_nodes, 1))

        # label 관련 내부 상태 (trainer가 set_label_info로 채움)
        self.label_init: torch.Tensor | None = None       # (N, C)
        self.train_mask: torch.Tensor | None = None       # (N,)
        self.hard_one_hot: torch.Tensor | None = None     # (N, C)

    # -------------------------------------------------------
    # Trainer에서 teacher 기반 label_init / mask 세팅용
    # -------------------------------------------------------
    def set_label_info(
        self,
        label_init: torch.Tensor,        # (N, C)
        train_mask: torch.Tensor,        # (N,)
        hard_one_hot: torch.Tensor,      # (N, C)
    ):
        device = next(self.parameters()).device
        self.label_init = label_init.to(device)
        self.train_mask = train_mask.to(device)
        self.hard_one_hot = hard_one_hot.to(device)

    # -------------------------------------------------------
    # Framework 규약: forward(x, edge_index)
    # -------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,             # (N, F)
        edge_index: torch.Tensor,    # (2, E)
    ) -> torch.Tensor:
        assert self.label_init is not None, "CPFStudent: label_init is not set. Call set_label_info() first."
        assert self.train_mask is not None, "CPFStudent: train_mask is not set. Call set_label_info() first."
        assert self.hard_one_hot is not None, "CPFStudent: hard_one_hot is not set. Call set_label_info() first."

        N = x.size(0)
        assert N == self.num_nodes, "num_nodes mismatch between data and CPFStudent"

        # 1) FT branch
        logits_ft = self.ft_mlp(x)           # (N, C)

        # 2) PLP branch (label propagation from label_init)
        plp = self.label_init                # (N, C)
        for _ in range(self.plp_steps):
            plp = self.plp_conv(plp, edge_index, num_nodes=N)
            # labeled node는 항상 ground truth 유지
            plp[self.train_mask] = self.hard_one_hot[self.train_mask]

        plp_logits = plp                     # (N, C)

        # 3) Node-wise gate
        alpha = torch.sigmoid(self.alpha)    # (N, 1)
        if alpha.size(0) != N:
            # 혹시 num_nodes와 실제 N이 어긋나면 최소 길이 기준으로 맞추기
            M = min(N, alpha.size(0))
            alpha = alpha[:M]
            logits_ft = logits_ft[:M]
            plp_logits = plp_logits[:M]

        logits = alpha * plp_logits + (1.0 - alpha) * logits_ft  # (N, C)
        return logits
