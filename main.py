import os
from data import GNNDataLoader
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class TinyGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# -------------------------------------------------
# 3️⃣ 동작 테스트
# -------------------------------------------------
if __name__ == "__main__":
    # 이름만 바꿔가며 확인 가능: "cora", "citeseer", "pubmed", "ogbn-arxiv"
    loader = GNNDataLoader(name="ogbn-products", use_sampler=True )
    data, split_idx, num_classes, sampler = loader.load()

    print(f"\n✅ Dataset: {loader.name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_features}, Classes: {num_classes}")

    # Planetoid와 OGB의 split 형태가 다르므로 출력 형식 분리
    if loader.name.startswith("ogbn-"):
        for key, idx in split_idx.items():
            print(f"  {key} size: {len(idx)}")
    else:
        for key, idx in split_idx.items():
            print(f"  {key} size: {idx.shape[0]}")

    # 초간이 모델 forward test
    model = TinyGCN(data.num_features, 32, num_classes)
    if sampler is None:
        # ----------------------------------------
        # Full-batch forward test
        # ----------------------------------------
        out = model(data.x, data.edge_index)
        print("Full-batch forward OK")
        print("Output shape :", out.shape)

    else:
        # ----------------------------------------
        # NeighborSampler batch forward test
        # ----------------------------------------
        for batch_size, n_id, adjs in sampler:
            # 1-hop adjacency만 쓰면됨 (GCN은 1-hop Conv)
            edge_index, _, _ = adjs[0]

            sub_x = data.x[n_id]
            out = model(sub_x, edge_index)
            print("Batch forward OK")
            print("Batch output shape :", out.shape)
            print("Seed nodes (prediction target):", batch_size)
            break  # 테스트니까 한 번만
