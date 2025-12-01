import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from experiment.train import BaseTrainer  # 우리가 만든 BaseTrainer
from experiment.analyze import Evaluator


class CETrainer(BaseTrainer):
    """
    [Concrete Strategy 1]
    표준 Cross-Entropy Loss로 훈련하는 트레이너.
    (e.g., pretrain, finetune, 또는 non-distill 벤치마킹용)

    이 트레이너는 GCN, GAT, SAGE, GIN 등 'model'의
    구체적인 아키텍처와 '무관하게' 작동한다.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 evaluator: Evaluator,
                 device: torch.device,
                 train_mask=None,
                 **kwargs):  # scheduler, logger 등을 받기 위함

        super().__init__(model, optimizer, evaluator, device, **kwargs)
        self.train_mask = train_mask

    def _compute_loss(self, batch: Data) -> tuple[torch.Tensor, dict]:
        # 1. Model Forward (GCN이든 GAT든 상관없이 logits 반환)
        logits = self.model(batch.x, batch.edge_index)

        # 2. Full-batch / Mini-batch 모드 자동 감지
        # --- ClusterLoader mini-batch ---
        if hasattr(batch, "n_id"):
            # cluster mode
            global_ids = batch.n_id
            global_mask = self.train_mask[global_ids]  # global mask slice
            target_nodes = torch.where(global_mask)[0]

            target_logits = logits[target_nodes]
            target_labels = batch.y[target_nodes].squeeze().long()
        elif hasattr(batch, 'batch_size'):
            # (Mini-batch) PyG NeighborLoader 표준
            # 0 ~ batch_size-1 까지가 타겟 노드
            target_logits = logits[:batch.batch_size]
            target_labels = batch.y[:batch.batch_size].squeeze().long()
        else:
            # (Full-batch) GNNDataLoader(v2)가 보장해준 'mask' 사용
            target_logits = logits[batch.train_mask]
            target_labels = batch.y[batch.train_mask].squeeze().long()

        # 3. Loss 계산
        loss = F.cross_entropy(target_logits, target_labels)

        log_dict = {"loss_ce": loss.item(), "loss_total": loss.item()}

        return loss, log_dict
