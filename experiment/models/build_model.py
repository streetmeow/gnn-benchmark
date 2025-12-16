# models/model_builder.py

from experiment.models.architectures import GCN, GAT, GraphSAGE, GIN, SGC


def build_model(cfg, in_dim, out_dim, model=None):
    """
    cfg.model 에 정의된 타입에 따라 적절한 GNN 모델을 생성.
    """
    m = model if model is not None else cfg.model
    name = m.name.lower()
    hidden = m.hidden_dim
    layers = m.num_layers
    dropout = m.dropout
    batchnorm = cfg.train.get("use_batchnorm", False)
    activation = m.get("activation", "relu")  # 기본 relu

    if name == "gcn":
        if batchnorm:
            print("⚠️ Warning: GCN with BatchNorm enabled.")
        return GCN(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation,
            use_batchnorm=batchnorm
        )

    elif name == "gat":
        heads = m.get("heads", 4)
        concat = m.get("concat", False)

        return GAT(
            in_dim=in_dim,
            hidden_dim=hidden,
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation,
            heads=heads,
            concat=concat,
            use_batchnorm=batchnorm
        )

    elif name == "graphsage":
        aggr = m.get("aggr", "mean")

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
    elif name == "sgc":
        return SGC(
            in_dim=in_dim,
            hidden_dim=hidden,  # 사실 사용 안 함
            out_dim=out_dim,
            num_layers=layers,
            dropout=dropout,
            activation=activation,
            use_batchnorm=batchnorm
        )

    else:
        raise ValueError(f"Unknown model type: {name}")
