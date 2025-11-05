
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_planetoid(name: str, root: str):
    ds = Planetoid(root=root, name=name.capitalize(), transform=NormalizeFeatures())
    data = ds[0]
    return dict(data=data, num_features=ds.num_features, num_classes=ds.num_classes, loaders=None)

def load_ogb_node(name: str, root: str):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name=name, root=root)
    split = ds.get_idx_split()
    data = ds[0]
    data.y = data.y.view(-1)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split["train"]] = True
    data.val_mask[split["valid"]] = True
    data.test_mask[split["test"]] = True
    num_features = data.num_features
    num_classes = int(data.y.max().item() + 1)
    return dict(data=data, num_features=num_features, num_classes=num_classes, loaders=None)

def build_dataset(cfg):
    if cfg.data.loader == "planetoid":
        if cfg.data.name not in {"cora", "citeseer", "pubmed"}:
            raise ValueError("Planetoid supports: cora | citeseer | pubmed")
        return load_planetoid(cfg.data.name, cfg.data.root)
    elif cfg.data.loader == "ogb":
        if cfg.data.name != "ogbn-arxiv":
            raise ValueError("OGB loader example supports only 'ogbn-arxiv' here.")
        return load_ogb_node(cfg.data.name, cfg.data.root)
    else:
        raise ValueError(f"Unknown loader: {cfg.data.loader}")
