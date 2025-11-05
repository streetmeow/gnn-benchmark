
from configs.base_config import BaseConfig, ModelConfig, DataConfig, TrainConfig, WandbConfig

def get_config():
    cfg = BaseConfig()
    cfg.model = ModelConfig(name="gcn", hidden_dim=256, num_layers=2, dropout=0.5)
    cfg.data = DataConfig(name="cora", loader="planetoid")
    cfg.train = TrainConfig(epochs=200, lr=0.01, weight_decay=5e-4, optimizer="adam")
    cfg.wandb = WandbConfig(enable=True, project="gnn-bench", tags=["cora","gcn"])
    cfg.log_dir = "runs/cora_gcn"
    return cfg
