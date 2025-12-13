# experiment/experiments/cpf_experiment.py

import torch
import torch.nn as nn

from scripts import BaseExperiment
from experiment.models.custom import CPFStudent
from experiment.train.trainers.cpf_trainer import CPFTrainer
from experiment.analyze import Evaluator, Metrics
from experiment.models import build_model   # 기존 teacher 불러오는 함수

import logging
log = logging.getLogger(__name__)


class CPFExperiment(BaseExperiment):
    """
    CPF Student Distillation을 수행하는 Experiment 클래스.

    [역할]
    - teacher 모델 로드 + checkpoint
    - teacher logits 생성 (full-batch)
    - CPFStudent 생성
    - Evaluator 생성
    - CPFTrainer 생성
    """

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

    def _build_models_and_evaluator(self):
        cfg = self.cfg
        data = self.data
        device = self.device

        num_classes = self.num_classes
        in_dim = data.num_features

        # -------------------------------------------------------------
        # 1) Teacher model 생성
        # -------------------------------------------------------------
        teacher = build_model(
            cfg,
            in_dim=in_dim,
            out_dim=num_classes,
        ).to(device)

        # Teacher checkpoint 필수
        ckpt_path = cfg.get("teacher_checkpoint", None)
        if ckpt_path is None:
            raise ValueError("CPFExperiment requires cfg.teacher_checkpoint")

        log.info(f"[CPF] Loading teacher checkpoint: {ckpt_path}")
        self._load_checkpoint(teacher, ckpt_path, strict=True)
        teacher.eval()

        # -------------------------------------------------------------
        # 2) Teacher logits (full batch)
        # -------------------------------------------------------------
        with torch.no_grad():
            teacher_logits = teacher(data.x, data.edge_index)
        teacher_logits = teacher_logits.detach()  # (N, C)

        # -------------------------------------------------------------
        # 3) CPFStudent 모델 생성
        # -------------------------------------------------------------
        hidden_dim = cfg.model.hidden_dim
        plp_steps = cfg.model.get("plp_steps", 10)
        dropout = cfg.model.get("dropout", 0.5)

        student = CPFStudent(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_classes,
            num_nodes=data.num_nodes,
            plp_steps=plp_steps,
            dropout=dropout,
        )

        self.model = student.to(device)

        # -------------------------------------------------------------
        # 4) Evaluator 구성 (student 성능 측정용)
        # -------------------------------------------------------------
        metrics = Metrics(
            metric_names=cfg.experiment.metrics,
            num_classes=num_classes,
        ).to(device)

        criterion_eval = nn.CrossEntropyLoss().to(device)
        self.evaluator = Evaluator(
            model=self.model,
            criterion=criterion_eval,
            metrics=metrics,
            device=device,
        )

        # trainer에서 사용할 teacher_logits 보관
        self._teacher_logits = teacher_logits

    # ------------------------------------------------------------------
    #  학습 루프 (BaseExperiment.run()에서 호출)
    # ------------------------------------------------------------------
    def _run_training(self):
        cfg = self.cfg
        device = self.device

        train_loader = self.train_loader  # [data] (full-batch)
        valid_loader = self.valid_loader
        train_mode = self.train_mode
        valid_mode = self.valid_mode

        # -------------------------
        # Optimizer
        # -------------------------
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        # -------------------------
        # Trainer 생성 (순수 KD)
        # -------------------------
        trainer = CPFTrainer(
            model=self.model,
            optimizer=optimizer,
            evaluator=self.evaluator,
            device=device,
            teacher_logits=self._teacher_logits,
            data_full=self.data,
            logger=self.logger,
            save_checkpoint=cfg.experiment.save_checkpoint,
            patience=cfg.train.patience,
            use_early_stopping=cfg.train.use_early_stopping,
        )

        # -------------------------
        # 학습 시작
        # -------------------------
        trainer.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=cfg.train.epochs,
            train_mode=train_mode,
            valid_mode=valid_mode,
        )
