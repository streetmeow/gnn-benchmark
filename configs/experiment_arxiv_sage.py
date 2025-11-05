
from configs.base_config import BaseConfig, ModelConfig, DataConfig, TrainConfig, WandbConfig

def get_config():
    cfg = BaseConfig()
    cfg.model = ModelConfig(name="graphsage", hidden_dim=256, num_layers=3, dropout=0.5, agg="mean")
    cfg.data = DataConfig(name="ogbn-arxiv", loader="ogb")
    cfg.train = TrainConfig(epochs=100, lr=0.01, weight_decay=5e-4, optimizer="adam")
    cfg.wandb = WandbConfig(enable=True, project="gnn-bench", tags=["ogb","arxiv","graphsage"])
    cfg.log_dir = "runs/arxiv_sage"
    return cfg
