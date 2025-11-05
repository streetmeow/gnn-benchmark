
from configs.base_config import BaseConfig, ModelConfig, DataConfig, TrainConfig, WandbConfig

def get_config():
    cfg = BaseConfig()
    cfg.model = ModelConfig(name="gat", hidden_dim=8, num_layers=2, dropout=0.6, heads=8, concat=True)
    cfg.data = DataConfig(name="cora", loader="planetoid")
    cfg.train = TrainConfig(epochs=200, lr=0.005, weight_decay=5e-4, optimizer="adam")
    cfg.wandb = WandbConfig(enable=True, project="gnn-bench", tags=["cora","gat"])
    cfg.log_dir = "runs/cora_gat"
    return cfg
