
import torch
from torch_geometric.datasets import Planetoid, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler
# from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
# from torch.serialization import add_safe_globals


class GNNDataLoader:
    def __init__(self, name: str, root: str = "./dataset", use_sampler: bool = False,
                 sampler_sizes: tuple = (15, 10, 5), sampler_batch_size: int = 1024, sampler_num_workers: int = 4):
        self.name = name.lower()
        self.root = root
        self.use_sampler = use_sampler
        self.sampler_sizes = sampler_sizes
        self.sampler_batch_size = sampler_batch_size
        self.sampler_num_workers = sampler_num_workers
        self.data = None
        self.split_idx = None
        self.num_classes = None
        self.sampler = None

    def load(self):
        if self.name in ["cora", "citeseer", "pubmed"]:
            self._load_planetoid()
        elif self.name == "actor":
            self._load_actor()
        elif self.name in ["ogbn-arxiv", "ogbn-products"]:
            self._load_ogbn_arxiv()
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        if self.use_sampler:
            self.get_sampler()
        return self.data, self.split_idx, self.num_classes, self.sampler

    def _load_planetoid(self):
        dataset = Planetoid(root=f"{self.root}/{self.name}", name=self.name.capitalize())
        data = dataset[0]
        splits = {
            "train": torch.where(data.train_mask)[0],
            "valid": torch.where(data.val_mask)[0],
            "test":  torch.where(data.test_mask)[0],
        }
        self.data = data
        self.split_idx = splits
        self.num_classes = dataset.num_classes

    def _load_ogbn_arxiv(self):
        # add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        dataset = PygNodePropPredDataset(name=self.name, root=self.root)
        self.data = dataset[0]
        split_idx = dataset.get_idx_split()
        self.split_idx = split_idx
        self.num_classes = dataset.num_classes

    def _load_actor(self):
        dataset = Actor(root=f"{self.root}/actor")
        data = dataset[0]

        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)

        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)

        self.split_idx = {
            "train": perm[:train_end],
            "valid": perm[train_end:val_end],
            "test": perm[val_end:]
        }
        self.data = data
        self.num_classes = dataset.num_classes

    def get_sampler(self):
        data = self.data
        split_idx = self.split_idx
        if "train" not in split_idx:
            raise ValueError("No training split available for sampler.")
        train_nodes = split_idx["train"]
        self.sampler = NeighborSampler(
            data.edge_index,
            node_idx=train_nodes,
            sizes=list(self.sampler_sizes),
            batch_size=self.sampler_batch_size,
            shuffle=True,
            num_workers=self.sampler_num_workers,
        )
