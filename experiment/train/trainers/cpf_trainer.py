# experiment/train/trainers/cpf_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from experiment.train import BaseTrainer
from experiment.analyze import Evaluator


class CPFTrainer(BaseTrainer):
    """
    CPF Student용 Trainer (pure KD 베이스라인).

    - Teacher logits로만 KD loss 계산
    - PLP에 들어갈 label_init과 hard one-hot labels를 내부에서 구성
    - BaseTrainer의 train_epoch / validate / run 로직 그대로 사용,
      _compute_loss만 distillation 형태로 override.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        evaluator: Evaluator,
        device: torch.device,
        teacher_logits: torch.Tensor,
        data_full: Data,
        **kwargs,
    ):
        """
        Args:
            model: CPFStudent
            optimizer: optimizer
            evaluator: Evaluator (student 평가용)
            device: torch.device
            teacher_logits: (N, C) teacher logits (full graph 기준)
            data_full: full-batch Data (x, edge_index, y, train/val/test mask 포함)
        """
        super().__init__(model, optimizer, evaluator, device, **kwargs)

        self.data_full = data_full

        # Teacher probs (no temperature; 원본 CPF와 동일하게 사용)
        self.teacher_logits = teacher_logits.to(device)        # (N, C)
        self.teacher_probs = F.softmax(self.teacher_logits, dim=-1)

        # GT one-hot labels
        y = data_full.y.view(-1).long().to(device)
        num_classes = int(self.teacher_logits.size(-1))
        self.hard_one_hot = F.one_hot(y, num_classes=num_classes).float()  # (N, C)

        self.train_mask = data_full.train_mask.to(device)      # (N,)

        # 초기 label_init: teacher_probs + labeled nodes는 one-hot로 덮어쓰기
        self.label_init = self.teacher_probs.clone()
        self.label_init[self.train_mask] = self.hard_one_hot[self.train_mask]


        # Student 모델에 label info 주입 (forward는 (x, edge_index)만 받도록)
        if hasattr(self.model, "set_label_info"):
            self.model.set_label_info(
                label_init=self.label_init,
                train_mask=self.train_mask,
                hard_one_hot=self.hard_one_hot,
            )

    def _compute_loss(self, batch: Data) -> tuple[torch.Tensor, dict]:
        """
        BaseTrainer.train_epoch에서 호출되는 손실 계산 함수.

        여기서는 full-batch만 지원한다고 가정 (Cornell/Texas/Planetoid 등).
        """
        # batch = batch.to(self.device)  # full-batch일 때도 안전하게

        # 1) Student forward (CPFStudent)
        logits = self.model(batch.x, batch.edge_index)  # (N, C)

        # 2) KD loss (unlabeled nodes 기준으로 KL)
        student_log_probs = F.log_softmax(logits, dim=-1)

        idx_no_train = ~self.train_mask  # True = unlabeled nodes
        kd_loss = F.mse_loss(
            logits[idx_no_train],
            self.teacher_probs[idx_no_train],
        )

        loss = kd_loss

        log_dict = {
            "loss_total": loss.item(),
            "loss_kd": kd_loss.item(),
        }

        return loss, log_dict
