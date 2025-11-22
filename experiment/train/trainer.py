# experiment/trainers/base_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from scripts import Logger
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Literal, Optional, Iterable
import torchmetrics
import logging

from experiment.analyze import Evaluator

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    훈련 로직의 '템플릿'을 제공하는 추상 기본 클래스.
    Resumable (학습 중단 후 재개) 기능이 포함됨.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, evaluator: Evaluator, device: torch.device,
                 scheduler: optim.lr_scheduler._LRScheduler = None, logger: Optional[Logger] = None,
                 save_checkpoint: bool = True):

        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.enable_checkpoint = save_checkpoint

        # 훈련 로깅을 위한 메트릭
        self.train_loss_metric = torchmetrics.MeanMetric().to(device)

        # 내부 상태
        self.best_metric_value = -1.0
        self.best_epoch = 0

        # [New] Resume을 위해 시작 에포크를 변수로 관리 (기본값 1)
        self.start_epoch = 1

    @abstractmethod
    def _compute_loss(self, batch: Data) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def train_epoch(self, loader: Iterable, mode: Literal["full", "mini"]):
        self.model.train()
        self.train_loss_metric.reset()

        for batch in tqdm(loader, desc="Training Epoch"):
            batch = batch.to(self.device)

            # 1. Loss 계산
            loss, log_dict = self._compute_loss(batch)

            # 2. 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 3. 기록
            if mode == "full":
                num_samples = batch.train_mask.sum().item()
            else:
                num_samples = batch.batch_size
            self.train_loss_metric.update(loss, weight=num_samples)

        return {"loss_avg": self.train_loss_metric.compute().item()}

    def validate(self, loader: Iterable, mode: Literal["full", "mini"], data_full: Optional[Data] = None):
        mask = None
        if mode == "full":
            if data_full is None:
                raise ValueError("valid_mode='full' but data_full was not provided")
            mask = data_full.val_mask

        metrics, _ = self.evaluator.evaluate(loader=loader, mode=mode, split_mask=mask)
        return metrics

    def run(self,
            train_loader: Iterable,
            valid_loader: Iterable,
            epochs: int,
            train_mode: Literal["full", "mini"],
            valid_mode: Literal["full", "mini"]):

        valid_data_full = valid_loader[0] if valid_mode == "full" else None

        # [Fix] 시작 에포크를 변수에서 가져옴 (Resume 시 1이 아닐 수 있음)
        for epoch in range(self.start_epoch, epochs + 1):

            # 1. Train
            train_metrics = self.train_epoch(train_loader, train_mode)

            # 2. Validate
            valid_metrics = self.validate(valid_loader, valid_mode, valid_data_full)

            # 3. Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['acc'])
                else:
                    self.scheduler.step()

            # 4. Logging
            log_data = {f"train/{k}": v for k, v in train_metrics.items()}
            log_data.update({f"valid/{k}": v for k, v in valid_metrics.items()})
            log_data["epoch"] = epoch
            log_data["lr"] = self.optimizer.param_groups[0]['lr']

            if self.logger:
                self.logger.log_epoch_metrics(log_data, step=epoch)

            log.info(
                f"Epoch: {epoch:03d} | Train Loss: {train_metrics['loss_avg']:.4f} | Valid Acc: {valid_metrics['acc']:.4f}")

            # 5. Checkpoint Saving (통합됨)
            current_metric = valid_metrics['acc']
            is_best = current_metric > self.best_metric_value

            if is_best:
                self.best_metric_value = current_metric
                self.best_epoch = epoch

            # [Fix] save_snapshot 하나로 통합
            self.save_snapshot(epoch, current_metric, is_best)

        log.info(f"Phase complete. Best Valid Acc ({self.best_metric_value:.4f}) at epoch {self.best_epoch}")

    def save_snapshot(self, epoch: int, metric_value: float, is_best: bool = False):
        """학습 상태 전체 저장"""

        snapshot = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric_value': self.best_metric_value,
            'best_epoch': self.best_epoch
        }

        save_dir = self.logger.output_dir if self.logger else "."

        # Last: 항상 저장 (복구용)
        torch.save(snapshot, os.path.join(save_dir, "last_checkpoint.pth"))

        # Best: 기록 갱신 시 저장 (결과용)
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "best_checkpoint.pth"))
            log.info(f"🏆 Best model saved at epoch {epoch} (Acc: {metric_value:.4f})")
        if self.enable_checkpoint:
            last_path = os.path.join(save_dir, "last_checkpoint.pth")
            torch.save(snapshot, last_path)

    def resume_checkpoint(self, path: str):
        """[New] 중단된 학습을 재개하기 위해 상태 로드"""
        if not os.path.exists(path):
            log.info(f"⚠️ Checkpoint not found at {path}. Starting from scratch.")
            return

        log.info(f"🔄 Resuming training from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        # 모델 & 옵티마이저 복구
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 상태 변수 복구
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric_value = checkpoint.get('best_metric_value', -1.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        log.info(f"✅ Resumed! Next epoch: {self.start_epoch}, Best Metric so far: {self.best_metric_value:.4f}")