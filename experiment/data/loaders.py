import torch
from torch_geometric.datasets import Planetoid, Actor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from omegaconf import DictConfig
import torch_geometric.transforms as T


class GNNDataLoader:
    """
    다양한 그래프 데이터셋을 로드하고 훈련/검증/테스트용 샘플러를 생성하는 클래스

    - 지원 데이터셋: Cora, Citeseer, Pubmed, Actor, OGBN-Arxiv, OGBN-Products
    - 샘플러가 필수인 미니배치 학습을 위해 NeighborLoader 사용
    - ogbn 데이터셋의 경우, 공식 제공되는 split 인덱스를 사용
    """
    def __init__(self, cfg_dataset):
        self.name = cfg_dataset.name.lower()
        self.root = cfg_dataset.root
        self.use_sampler = cfg_dataset.use_sampler

        self.data = None
        self.num_classes = None

    @staticmethod
    def idx_to_mask(idx: torch.Tensor, size: int) -> torch.Tensor:
        """
        train, val, test 인덱스에 대해 대상이 아닌 걸 False로 마스킹하는 함수
        """
        mask = torch.zeros(size, dtype=torch.bool)
        mask[idx] = True
        return mask

    def load(self):
        """데이터셋 로드 분기 처리"""
        if self.name in ["cora", "citeseer", "pubmed"]:
            self._load_planetoid()
        elif self.name == "actor":
            self._load_actor()
        elif self.name in ["ogbn-arxiv", "ogbn-products"]:
            self._load_ogbn()
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

        return self.data, self.num_classes

    def _load_planetoid(self):
        """
        Planetoid 데이터셋 로드 (Cora, Citeseer, Pubmed)
        """
        dataset = Planetoid(root=f"{self.root}/{self.name}", name=self.name.capitalize())
        data = dataset[0]
        self.data = data
        self.num_classes = dataset.num_classes

    def _load_ogbn(self):
        """
        OGBN 데이터셋 로드 (OGBN-Arxiv, OGBN-Products)
        """
        transform = None
        if self.name == "ogbn-arxiv":
            transform = T.Compose([T.ToUndirected()])
        dataset = PygNodePropPredDataset(name=self.name, root=self.root, transform=transform)
        data = dataset[0]
        split_idx = dataset.get_idx_split()

        # 'idx_to_mask' (정적 메서드)를 호출
        data.train_mask = self.idx_to_mask(split_idx["train"], data.num_nodes)
        data.val_mask = self.idx_to_mask(split_idx["valid"], data.num_nodes)
        data.test_mask = self.idx_to_mask(split_idx["test"], data.num_nodes)

        self.data = data
        self.num_classes = dataset.num_classes

    def _load_actor(self):
        """
        Actor 데이터셋 로드
        """
        dataset = Actor(root=f"{self.root}/actor")
        data = dataset[0]
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)

        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)

        data.train_mask = self.idx_to_mask(perm[:train_end], num_nodes)
        data.val_mask = self.idx_to_mask(perm[train_end:val_end], num_nodes)
        data.test_mask = self.idx_to_mask(perm[val_end:], num_nodes)

        self.data = data
        self.num_classes = dataset.num_classes

    def get_train_sampler(self, cfg_sampler: DictConfig, sampler_size) -> NeighborLoader:
        """훈련용 (train_mask) 샘플러 생성 (셔플 O)"""
        train_nodes = torch.where(self.data.train_mask)[0]
        return NeighborLoader(
            self.data,
            input_nodes=train_nodes,
            num_neighbors=list(sampler_size),   # GCN/GIN에 따라 리스트가 달라짐
            batch_size=cfg_sampler.batch_size,
            shuffle=cfg_sampler.get("shuffle", True),
            num_workers=cfg_sampler.get("num_workers", 0)
        )

    def get_valid_sampler(self, cfg_sampler: DictConfig, sampler_size) -> NeighborLoader:
        """검증용 (val_mask) 샘플러 생성 (셔플 X)"""
        valid_nodes = torch.where(self.data.val_mask)[0]
        return NeighborLoader(
            self.data,
            input_nodes=valid_nodes,
            num_neighbors=list(sampler_size),
            batch_size=cfg_sampler.batch_size,
            shuffle=False,
            num_workers=cfg_sampler.get("num_workers", 0)
        )

    def get_test_sampler(self, cfg_sampler: DictConfig, sampler_size) -> NeighborLoader:
        """테스트용 (test_mask) 샘플러 생성 (셔플 X)"""
        test_nodes = torch.where(self.data.test_mask)[0]
        return NeighborLoader(
            self.data,
            input_nodes=test_nodes,
            num_neighbors=list(sampler_size),
            batch_size=cfg_sampler.batch_size,
            shuffle=False,
            num_workers=cfg_sampler.get("num_workers", 0)
        )