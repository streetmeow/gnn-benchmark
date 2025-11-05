
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    optimizer: str = "adam"
    batch_size: Optional[int] = None  # full-batch by default

@dataclass
class ModelConfig:
    name: str = "gcn"    # "gcn" | "gat" | "graphsage"
    in_dim: int = -1     # auto-filled from dataset if <= 0
    hidden_dim: int = 256
    out_dim: int = -1    # auto-filled from dataset if <= 0
    num_layers: int = 2
    dropout: float = 0.5
    # for GAT
    heads: int = 8
    concat: bool = True
    # for GraphSAGE
    agg: str = "mean"    # "mean" | "max" | "sum"

@dataclass
class DataConfig:
    name: str = "cora"         # "cora" | "citeseer" | "pubmed" | "ogbn-arxiv"
    root: str = "data/"
    loader: str = "planetoid"  # "planetoid" | "ogb"
    sampling: str = "full"     # available hook for extension

@dataclass
class WandbConfig:
    enable: bool = True                 # if False, force disable
    project: str = "gnn-bench"
    entity: Optional[str] = None        # optional
    run_name: Optional[str] = None      # optional; auto if None
    mode: Optional[str] = None          # "online" | "offline" | "disabled" | None -> auto
    tags: List[str] = field(default_factory=list)

@dataclass
class BaseConfig:
    seed: int = 42
    device: str = "cuda"                # or "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_dir: str = "runs/"
