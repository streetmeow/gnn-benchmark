
import torch
from data.loaders import build_dataset
from models.zoo import gcn, gat, graphsage
from utils.logger import get_logger
from utils.seed import fix_seed

def build_all(cfg):
    logger = get_logger()
    fix_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    bundle = build_dataset(cfg)
    cfg.model.in_dim = bundle["num_features"] if cfg.model.in_dim <= 0 else cfg.model.in_dim
    cfg.model.out_dim = bundle["num_classes"] if cfg.model.out_dim <= 0 else cfg.model.out_dim

    if cfg.model.name == "gcn":
        model = gcn.build_from_cfg(cfg, bundle["data"]).to(device)
    elif cfg.model.name == "gat":
        model = gat.build_from_cfg(cfg, bundle["data"]).to(device)
    elif cfg.model.name == "graphsage":
        model = graphsage.build_from_cfg(cfg, bundle["data"]).to(device)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    if cfg.train.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    return dict(cfg=cfg, logger=logger, model=model, device=device, data_bundle=bundle, optimizer=optimizer, criterion=criterion)
