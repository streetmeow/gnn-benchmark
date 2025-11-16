# models/model_builder.py

from experiment.models.architectures import GCN, GAT, GraphSAGE, GIN


def build_model(cfg, in_dim, out_dim):
    """
    cfg.model 에 정의된 타입에 따라 적절한 GNN 모델을 생성.

    공용 interface:
        model = build_model(cfg, in_dim, out_dim)
        out = model(x, edge_index)

    Args:
        cfg: Hydra configuration (cfg.model.*)
        in_dim: 입력 feature dimension
        out_dim: 클래스 개수 (node classification 기준)

    Returns:
        nn.Module (BaseGNN 상속 모델)
    """

    name = cfg.model.name.lower()
    hidden = cfg.model.hidden_dim
    layers = cfg.model.num_layers
    dropout = cfg.model.dropout
    activation = cfg.model.get("activation", "relu")  # 기본 relu

    if name == "gcn":
        return GCN(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation
        )

    elif name == "gat":
        heads = cfg.model.get("heads", 4)
        concat = cfg.model.get("concat", False)

        return GAT(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation,
            heads=heads,
            concat=concat
        )

    elif name == "graphsage":
        aggr = cfg.model.get("aggr", "mean")

        return GraphSAGE(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation,
            aggr=aggr
        )

    elif name == "gin":
        return GIN(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation
        )

    else:
        raise ValueError(f"Unknown model type: {name}")
