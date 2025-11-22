# experiment/base_experiment.py
import copy

import torch
import torch.nn as nn
import os
import logging
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from thop import profile

# --- 우리가 만든 모듈 임포트 ---
from experiment.data import GNNDataLoader
from experiment.analyze import Evaluator
from .logger import Logger
from .utils import FeatureExtractor

log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """
    실험 진행을 전체적으로 관리하는 추상 클래스
    """

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

        # 2. 모델 컨테이너 초기화 (순서 중요: 빌드 전에 그릇이 있어야 함)
        self.model: nn.Module = None
        self.evaluator: Evaluator = None
        self.helper_models = nn.ModuleDict()  # [OK] 위치 아주 좋습니다.

        # 3. 추상 로직: 모델 빌드
        self._build_models_and_evaluator()

        if self.model is None or self.evaluator is None:
            raise NotImplementedError(
                "Child class must set 'self.model' and 'self.evaluator' in '_build_models_and_evaluator'"
            )

        # 4. [권장 수정] GPU 이동 -> 가중치 로드 순서가 더 안전합니다.
        self.model = self.model.to(self.device)
        self.helper_models = self.helper_models.to(self.device)

        # 5. 가중치 로드 (파인튜닝 or Distill 초기화)
        self._load_initial_weights()

    def _setup_device(self) -> torch.device:
        device = torch.device(f"cuda:{self.cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        return device

    def _load_data(self) -> tuple:
        loader = GNNDataLoader(self.cfg.dataset)
        data, num_classes = loader.load()
        data = data.to(self.device)

        if not self.cfg.dataset.use_sampler:
            train_loader = valid_loader = test_loader = [data]
            train_mode = valid_mode = test_mode = "full"
        else:
            log.info("Building samplers for train, valid, test...")
            train_loader = loader.get_train_sampler(self.cfg.sampler)
            valid_loader = loader.get_valid_sampler(self.cfg.sampler)
            test_loader = loader.get_test_sampler(self.cfg.sampler)
            train_mode = valid_mode = test_mode = "mini"

        log.info(f"Data loaded. Train mode: {train_mode}, Num classes: {num_classes}")
        return data, num_classes, train_loader, valid_loader, test_loader, train_mode, valid_mode, test_mode

    def _load_checkpoint(self, model: nn.Module, path: str, strict: bool = True):
        """가중치 로드 헬퍼 (포장지 뜯기 기능 포함)"""
        if not os.path.exists(path):
            log.warning(f"Checkpoint path '{path}' does not exist. Skipping load.")
            return

        log.info(f"Loading checkpoint from '{path}'")
        # map_location으로 디바이스 맞춰서 로드
        state_dict = torch.load(path, map_location=self.device)

        # [중요] 스냅샷 딕셔너리인 경우 'model_state_dict'만 추출
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        try:
            model.load_state_dict(state_dict, strict=strict)
            log.info("Checkpoint loaded successfully.")
        except Exception as e:
            log.error(f"Error loading checkpoint: {e}")
            raise e

    def _load_initial_weights(self):
        """Config 설정에 따라 초기 가중치 로드"""
        if self.cfg.get("checkpoint_path"):
            log.info("Detecting 'checkpoint_path' in config. Loading initial weights...")
            self._load_checkpoint(self.model, self.cfg.checkpoint_path)

        if self.cfg.get("helper_checkpoints"):
            log.info("Detecting 'helper_checkpoints' in config. Loading helper model weights...")
            for name, path in self.cfg.helper_checkpoints.items():
                if name in self.helper_models:
                    self._load_checkpoint(self.helper_models[name], path)
                else:
                    log.warning(f"Helper model '{name}' not found in 'self.helper_models'. Skipping load.")

    # --- 3. 추상 로직 ---

    @abstractmethod
    def _build_models_and_evaluator(self):
        raise NotImplementedError

    @abstractmethod
    def _run_training(self):
        raise NotImplementedError

    # --- 4. 공통 로직 ---

    def _profile_model_complexity(self):
        log.info("[Profiling] Calculating model complexity (FLOPs and Params)...")
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        try:
            total_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
            sample_batch = next(iter(self.train_loader))
            sample_batch = sample_batch.to(self.device)
            macs, _ = profile(model_copy, inputs=(sample_batch.x, sample_batch.edge_index), verbose=False)
            flops = macs * 2  # FLOPs 계산
            complexity = {
                "params": total_params,
                "flops": flops,
                "params_million": total_params / 1e6,
                "flops_giga": flops / 1e9
            }
            log.info(f"- params: {complexity['params']} ({complexity['params_million']:.2f} M)")
            log.info(f"- flops: {complexity['flops']} ({complexity['flops_giga']:.2f} G)")
            if hasattr(self.logger, 'save_model_complexity'):
                self.logger.save_model_complexity(complexity)
        except Exception as e:
            log.error(f"Failed to profile model complexity: {e}")
            log.error(f"Skipping model complexity profiling.")
        finally:
            del model_copy

    def _run_final_test(self):
        log.info("--- 🏁 All training complete. Loading best model for final test. ---")

        # BaseTrainer가 저장하는 이름과 일치시킴 (best_checkpoint.pth)
        best_model_path = os.path.join(self.logger.output_dir, "best_checkpoint.pth")
        last_layer_tensor = None

        # 직접 로드하지 않고 _load_checkpoint 재활용
        # (이렇게 해야 snapshot 구조일 때 'model_state_dict' 키를 자동으로 처리함)
        try:
            self._load_checkpoint(self.model, best_model_path)
        except Exception as e:
            log.warning(f"Could not load best model ({e}). Testing with current model state.")

        visualize = self.cfg.experiment.get("visualize", False)
        target_layers = self.cfg.experiment.get("feature_extractor_layers", [])

        if target_layers:
            log.info(f"Extracting features from layers: {target_layers}")
            
            with FeatureExtractor(self.model, target_layers) as extractor:
                self.model.eval()
                with torch.no_grad():
                    self.model(self.data.x, self.data.edge_index)
                extracted_features = extractor.features
            test_results, _ = self.evaluator.evaluate(
                loader=self.test_loader,
                mode=self.test_mode,
                split_mask=self.data.test_mask if self.test_mode == "full" else None,
                return_logits=False
            )

            if extracted_features:
                full_tensor = extracted_features[target_layers[-1]]
                last_layer_tensor = full_tensor[self.data.test_mask]
            save_path = os.path.join(self.logger.output_dir, "extracted_embeddings.pt")
            torch.save(extracted_features, save_path)
            log.info(f"Extracted features saved to {save_path}")
        else:
            test_results, last_layer_tensor = self.evaluator.evaluate(
                loader=self.test_loader,
                mode=self.test_mode,
                split_mask=self.data.test_mask if self.test_mode == "full" else None,
                return_logits=visualize
            )
        if visualize and last_layer_tensor is not None:
            self._visualize_embeddings(last_layer_tensor, self.data.y[self.data.test_mask])

        log.info(f"Final Test Results: {test_results}")
        self.logger.save_final_results(test_results)

    def _visualize_embeddings(self, embeddings: torch.Tensor, labels: torch.Tensor):
        try:
            import umap
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            log.warning("UMAP or Matplotlib not installed. Skipping embedding visualization.")
            return

        log.info("[Visualization] Generating UMAP plot for embeddings...")
        x = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        if y.ndim > 1:
            y = y.flatten()
        max_sample_num = 10000
        if x.shape[0] > max_sample_num:
            log.info(f"Subsampling {max_sample_num} points from {x.shape[0]} for UMAP visualization...")
            indices = np.random.choice(x.shape[0], max_sample_num, replace=False)
            x = x[indices]
            y = y[indices]
        try:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', n_jobs=1)
            embedding_2d = reducer.fit_transform(x)

            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=y,
                cmap='tab10',
                s=5,
                alpha=0.7
            )
            plt.colorbar(scatter, ticks=range(len(np.unique(y))))
            plt.title(f"{self.cfg.dataset.name} -  {self.cfg.model.name} Embedding UMAP")
            plt.axis('off')
            self.logger.save_visualization(plt)
        except Exception as e:
            log.error(f"Failed to generate UMAP visualization: {e}")

    def run(self):
        log.info(f"--- 🚀 Starting Experiment: {self.__class__.__name__} ---")
        # 모델 복잡도 프로파일링
        self._profile_model_complexity()
        # 1. 훈련
        mode = self.cfg.experiment.get("mode", "train")
        if mode == "train":
            log.info("Starting training mode...")
            self._run_training()
        else:
            log.info("Skipping training as per config. Proceeding to final test...")

        # 2. 최종 테스트
        self._run_final_test()

        log.info(f"--- ✅ Experiment Finished: {self.__class__.__name__} ---")
