# experiment/trainers/base_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from scripts import Logger
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Literal, Optional, Iterable
import torchmetrics

from experiment.analyze import Evaluator  # 우리가 만든 Evaluator


# from experiment.logger import BaseLogger # (추후 WandB/Local 로거)

class BaseTrainer(ABC):
    """
    훈련 로직의 '템플릿'을 제공하는 추상 기본 클래스.

    공통 로직 (epoch 루프, validation 호출, 옵티마이저 스텝, 로깅, 체크포인트)은
    이 클래스가 처리하고, 'Loss 계산'이라는 핵심 로직만
    하위 클래스(CETrainer, KDTrainer 등)가 오버라이드하여 채운다.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, evaluator: Evaluator, device: torch.device,
                 scheduler: optim.lr_scheduler._LRScheduler = None, logger: Optional[Logger] = None):

        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator  # (Valid/Test용)
        self.device = device
        self.scheduler = scheduler
        self.logger = logger  # (WandB 연동 지점)

        # 훈련 로깅을 위한 메트릭 (배치별 평균)
        self.train_loss_metric = torchmetrics.MeanMetric().to(device)

        # Early stopping / Best model 저장을 위한 내부 상태
        self.best_metric_value = -1.0  # (Acc/F1 기준, 높을수록 좋음)
        self.best_epoch = 0

    @abstractmethod
    def _compute_loss(self, batch: Data) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def train_epoch(self, loader: Iterable, mode: Literal["full", "mini"]):
        self.model.train()
        self.train_loss_metric.reset()

        for batch in tqdm(loader, desc="Training Epoch"):
            batch = batch.to(self.device)

            # 1. [핵심] Loss 계산 (하위 클래스에 위임)
            loss, log_dict = self._compute_loss(batch)

            # 2. 역전파 (공통 로직)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 3. 훈련 Loss 누적 (가중 평균)
            # 풀배치면 train_mask.sum(), 미니배치면 batch_size
            if mode == "full":
                # 'train_mask' (bool)에서 True 개수를 세어 샘플 수로 사용
                num_samples = batch.train_mask.sum().item()
            else:
                num_samples = batch.batch_size

            self.train_loss_metric.update(loss, weight=num_samples)

        # 1 에폭의 평균 Loss 반환
        return {"loss_avg": self.train_loss_metric.compute().item()}

    def validate(self, loader: Iterable, mode: Literal["full", "mini"], data_full: Optional[Data] = None):
        """
        검증(Validation) 수행. Evaluator를 호출하는 래퍼(wrapper).
        """
        mask = None
        if mode == "full":
            if data_full is None:
                raise ValueError("valid_mode='full' but data_full was not provided")
            # GNNDataLoader(v2)가 data.val_mask (bool)를 생성했음을 전제
            mask = data_full.val_mask

        return self.evaluator.evaluate(
            loader=loader,
            mode=mode,
            split_mask=mask
        )

    def run(self,
            train_loader: Iterable,
            valid_loader: Iterable,
            epochs: int,
            train_mode: Literal["full", "mini"],
            valid_mode: Literal["full", "mini"]):
        """
        메인 훈련 루프 (템플릿).
        main.py가 Hydra 'phase'별로 이 함수를 호출.
        """

        # 'full' 모드일 경우, loader는 [data] 형태이므로 data 객체를 미리 추출
        valid_data_full = valid_loader[0] if valid_mode == "full" else None

        for epoch in range(1, epochs + 1):

            # 1. Train
            train_metrics = self.train_epoch(train_loader, train_mode)

            # 2. Validate
            valid_metrics = self.validate(valid_loader, valid_mode, valid_data_full)

            # 3. LR Scheduler Step
            if self.scheduler:
                self.scheduler.step(valid_metrics['acc'])

            log_data = {f"train/{k}": v for k, v in train_metrics.items()}
            log_data.update({f"valid/{k}": v for k, v in valid_metrics.items()})
            log_data["epoch"] = epoch
            log_data["lr"] = self.optimizer.param_groups[0]['lr']

            if self.logger:
                self.logger.log_epoch_metrics(log_data, step=epoch)

            print(
                f"Epoch: {epoch:03d} | Train Loss: {train_metrics['loss_avg']:.4f} | Valid Acc: {valid_metrics['acc']:.4f}")

            # 5. Checkpoint (Best Model Saving)
            # (valid_metrics['acc']를 기준으로 저장한다고 가정)
            current_metric = valid_metrics['acc']
            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                self._save_checkpoint("best_model.pth")

        print(f"Phase complete. Best Valid Acc ({self.best_metric_value:.4f}) at epoch {self.best_epoch}")

    def _save_checkpoint(self, path: str):
        """(임시) 모델 체크포인트 저장"""
        if self.logger:
            self.logger.save_checkpoint(self.model, filename=path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")
