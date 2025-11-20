# experiment/base_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import DictConfig
import logging
from abc import ABC, abstractmethod

# --- 우리가 만든 모듈 임포트 ---
# (파일 경로 '..'는 네 프로젝트 구조에 맞게 조정해야 할 수 있음)
from experiment.data import GNNDataLoader
from experiment.analyze import Evaluator
from .logger import Logger
import os

log = logging.getLogger(__name__)


class BaseExperiment(ABC):

    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        self.device = self._setup_device()

        # 1. 공통 로직: 데이터 로드
        (
            self.data,
            self.num_classes,
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.train_mode,
            self.valid_mode,
            self.test_mode
        ) = self._load_data()

        # 2. 추상 로직: 모델/평가자 빌드
        # (하위 클래스가 이 메서드를 실행하여 self.student_model과
        # self.evaluator를 '반드시' 설정해야 함)
        self.student_model: nn.Module = None
        self.evaluator: Evaluator = None
        self._build_models_and_evaluator()

        if self.student_model is None or self.evaluator is None:
            raise NotImplementedError(
                "Child class must set 'self.student_model' and 'self.evaluator' in '_build_models_and_evaluator'"
            )

    def _setup_device(self) -> torch.device:
        """1. 환경 설정: 디바이스 설정"""
        device = torch.device(f"cuda:{self.cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        return device

    def _load_data(self) -> tuple:
        """2. 공통 로직: 데이터 로드 (v2.3)"""
        loader = GNNDataLoader(self.cfg.dataset)
        data, num_classes = loader.load()
        data = data.to(self.device)

        if not self.cfg.dataset.use_sampler:  # Full-batch
            train_loader = valid_loader = test_loader = [data]
            train_mode = valid_mode = test_mode = "full"
        else:  # Mini-batch
            log.info("Building samplers for train, valid, test...")
            # 'cfg.sampler' (e.g., gcn_2layer.yaml)를 loader의 'get' 메서드에 주입
            train_loader = loader.get_train_sampler(self.cfg.sampler)
            valid_loader = loader.get_valid_sampler(self.cfg.sampler)
            test_loader = loader.get_test_sampler(self.cfg.sampler)
            train_mode = valid_mode = test_mode = "mini"

        log.info(f"Data loaded. Train mode: {train_mode}, Num classes: {num_classes}")
        return data, num_classes, train_loader, valid_loader, test_loader, train_mode, valid_mode, test_mode

    # --- 3. 추상 로직 (하위 클래스가 구현) ---

    @abstractmethod
    def _build_models_and_evaluator(self):
        raise NotImplementedError

    @abstractmethod
    def _run_training(self):
        raise NotImplementedError

    # --- 4. 공통 로직 (템플릿) ---

    def _run_final_test(self):
        log.info("--- 🏁 All training complete. Loading best model for final test. ---")
        best_model_path = os.path.join(self.logger.output_dir, "best_model.pth")

        try:
            self.student_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        except FileNotFoundError:
            log.warning("Could not find 'best_model.pth'. Testing with current model state.")

        test_results = self.evaluator.evaluate(
            loader=self.test_loader,
            mode=self.test_mode,
            split_mask=self.data.test_mask if self.test_mode == "full" else None
        )

        log.info(f"Final Test Results: {test_results}")
        self.logger.save_final_results(test_results)

    def run(self):
        log.info(f"--- 🚀 Starting Experiment: {self.__class__.__name__} ---")

        # 1. 훈련 실행 (하위 클래스의 '전략'을 호출)
        self._run_training()

        # 2. 최종 테스트 (공통 로직 호출)
        self._run_final_test()

        log.info(f"--- ✅ Experiment Finished: {self.__class__.__name__} ---")