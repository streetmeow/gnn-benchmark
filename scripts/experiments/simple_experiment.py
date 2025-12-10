# experiment/simple_experiment.py

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from omegaconf import DictConfig

# --- '현장 감독' 템플릿과 '작업자' 임포트 ---
from scripts import BaseExperiment
from experiment.train import CETrainer
from experiment.models import build_model
from experiment.analyze import Metrics, Evaluator
import torch.nn as nn

log = logging.getLogger(__name__)


class SimpleExperiment(BaseExperiment):
    def __init__(self, cfg: DictConfig, logger):
        super().__init__(cfg, logger)

    def _build_models_and_evaluator(self):
        cfg = self.cfg

        # 1. 단일 모델 빌드 (GCN, GAT, GIN, SAGE...)
        self.model = build_model(
            self.cfg,  # ⬅️ 'cfg.model' (단일 모델 설정)
            self.data.num_features,
            self.num_classes
        ).to(self.device)

        log.info(f"Built Single Model ({cfg.model.name}):\n{self.model}")

        # 2. 평가자 빌드 (단일 모델 기준)
        metrics = Metrics(
            metric_names=cfg.experiment.metrics,  # (config에서 읽어오도록 수정)
            num_classes=self.num_classes
        ).to(self.device)

        criterion_eval = nn.CrossEntropyLoss().to(self.device)

        self.evaluator = Evaluator(
            model=self.model,
            criterion=criterion_eval,
            metrics=metrics,
            device=self.device
        )

    def _run_training(self):
        cfg = self.cfg
        log.info(f"--- 🚀 Starting Simple Training (CETrainer only) ---")

        # 'cfg.train' (top-level)에서 하이퍼파라미터 읽기
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )

        scheduler = None
        if cfg.train.get("use_scheduler", False):
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # '작업자'로 CETrainer를 고정
        trainer = CETrainer(
            model=self.model,
            optimizer=optimizer,
            evaluator=self.evaluator,
            device=self.device,
            scheduler=scheduler,
            logger=self.logger,
            save_checkpoint=cfg.experiment.save_checkpoint,
            patience=cfg.train.get("patience", 100),
            use_early_stopping=cfg.train.get("use_early_stopping", True)
        )

        # BaseTrainer의 공통 'run' 메서드 호출
        trainer.run(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            epochs=self.cfg.train.epochs,  # top-level epochs
            train_mode=self.train_mode,
            valid_mode=self.valid_mode
        )