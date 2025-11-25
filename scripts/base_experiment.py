import copy
import torch
import torch.nn as nn
import os
import logging
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from thop import profile
from tqdm import tqdm  # 진행률 표시용

# --- 우리가 만든 모듈 임포트 ---
from experiment.data import GNNDataLoader
from experiment.analyze import Evaluator
from .logger import Logger

# FeatureExtractor 클래스는 더 이상 필요 없음 (직접 구현)

log = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """
    실험 진행을 전체적으로 관리하는 추상 클래스

    [담당 작업]
    1. 데이터 로드
    2. 모델 및 평가자 지정 (자식 클래스에서 구현)
    3. 디바이스 설정 및 체크포인트 로드
    4. 모델 복잡도 프로파일링
    5. 훈련 루프 실행 (자식 클래스에서 구현)
    6. 최종 테스트 및 시각화
    7. 중간 레이어 Feature Extraction 지원

    자식 클래스는 반드시 '_build_models_and_evaluator'와 '_run_training' 메서드를 구현해야 함.
    """

    def __init__(self, cfg: DictConfig, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        self.device = self._setup_device()

        # 1. 데이터 로드
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

        # 2. 모델 및 평가자 초기화
        self.model: nn.Module = None
        self.evaluator: Evaluator = None
        self.helper_models = nn.ModuleDict()

        self._build_models_and_evaluator()

        if self.model is None or self.evaluator is None:
            raise NotImplementedError("Child class must set 'self.model' and 'self.evaluator'")

        # 3. 디바이스 이동 및 가중치 로드
        self.model = self.model.to(self.device)
        self.helper_models = self.helper_models.to(self.device)
        self._load_initial_weights()

    # --- [Helper Methods] ---
    def _setup_device(self) -> torch.device:
        device = torch.device(f"cuda:{self.cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        return device

    def _load_data(self) -> tuple:
        """
        데이터 로드 및 샘플러 연결
        """
        loader = GNNDataLoader(self.cfg.dataset)
        # 데이터 및 클래스 수 로드
        data, num_classes = loader.load()
        data = data.to(self.device)

        if not self.cfg.dataset.use_sampler:
            # 샘플러 미사용 시, 풀배치 모드로 처리
            return data, num_classes, [data], [data], [data], "full", "full", "full"
        else:
            # 샘플러 사용 시, 미니배치 모드로 처리
            # loader 에서 sampler 생성 메서드 호출
            log.info("Building samplers...")
            return (data, num_classes,
                    loader.get_train_sampler(self.cfg.sampler, self.cfg.model.sizes),
                    loader.get_valid_sampler(self.cfg.sampler, self.cfg.model.sizes),
                    loader.get_test_sampler(self.cfg.sampler, self.cfg.model.sizes),
                    "mini", "mini", "mini")

    def _load_checkpoint(self, model: nn.Module, path: str, strict: bool = True):
        """
        체크포인트 로드 시도
        """
        if not os.path.exists(path):
            log.warning(f"Checkpoint path '{path}' does not exist. Skipping load.")
            return

        log.info(f"Loading checkpoint from '{path}'")
        # 모델을 디바이스에 맞게 로드
        state_dict = torch.load(path, map_location=self.device)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # thop 에서 불필요한 값이 발생하였을 경우 제거
        # 레거시임. 현재는 관련 코드 deepcopy 기반으로 동작함
        for k in list(state_dict.keys()):
            if "total_ops" in k or "total_params" in k:
                del state_dict[k]

        try:
            # 매핑해서 값을 넣음
            model.load_state_dict(state_dict, strict=strict)
            log.info("Checkpoint loaded successfully.")
        except Exception as e:
            log.error(f"Error loading checkpoint: {e}")
            raise e

    def _load_initial_weights(self):
        """
        초기 체크포인트 로드 (메인 모델 및 보조용 모델)
        """

        # 학습을 이어서 할 경우 모델 체크포인트 로드
        if self.cfg.get("checkpoint_path"):
            self._load_checkpoint(self.model, self.cfg.checkpoint_path)

        # 보조 모델 체크포인트 로드
        # 단, 보조 모델이 self.helper_models에 존재할 때만 로드
        # 모델 생성은 '_build_models_and_evaluator'에서 수행됨
        if self.cfg.get("helper_checkpoints"):
            for name, path in self.cfg.helper_checkpoints.items():
                if name in self.helper_models:
                    self._load_checkpoint(self.helper_models[name], path)

    # --- [Abstract Methods] ---
    @abstractmethod
    def _build_models_and_evaluator(self):
        """
        자식 클래스에서 구현
        - 모델을 만들고 self.model에 할당
        - teacher 모델이나 평가자도 이곳에서 생성
        """
        raise NotImplementedError

    @abstractmethod
    def _run_training(self):
        raise NotImplementedError

    # --- [Core Logic: Complexity] ---
    def _profile_model_complexity(self):
        """
        모델 파라미터 수 및 FLOPs 계산
        1. 모델 복사본 생성 (원본 보호용)
        2. 샘플 배치로 프로파일링 수행
        3. 결과 로깅 및 저장
        4. 복사본 삭제
        """
        log.info("[Profiling] Calculating model complexity...")

        # 모델 복사본 생성
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        try:
            # 파라미터 수, FLOPs 계산
            total_params = sum(p.numel() for p in model_copy.parameters() if p.requires_grad)
            sample_batch = next(iter(self.train_loader))
            sample_batch = sample_batch.to(self.device)
            macs, _ = profile(model_copy, inputs=(sample_batch.x, sample_batch.edge_index), verbose=False)

            complexity = {
                "params": total_params, "flops": macs * 2,
                "params_million": total_params / 1e6, "flops_giga": (macs * 2) / 1e9
            }
            log.info(f"- params: {complexity['params_million']:.2f} M")
            log.info(f"- flops: {complexity['flops_giga']:.2f} G")

            # 로거에 저장
            if hasattr(self.logger, 'save_model_complexity'):
                self.logger.save_model_complexity(complexity)
        except Exception as e:
            log.error(f"Failed to profile complexity: {e}")
        finally:
            # 복사본 삭제
            del model_copy

    # --- [Core Logic: Feature Extraction] ---
    def _extract_features(self, target_layers: list) -> dict:
        """
        지정된 레이어에서 중간 embedding 값 추출 (Distillation 에 사용하기 위함임)
        1. 포워드 훅 등록
        2. 데이터 로더 순회하며 추출
        3. 결과 반환

        가상의 루프를 돌면서 각 배치마다 포워드 훅이 작동하여 중간 출력을 수집함.
        """
        log.info(f"Extracting features from layers: {target_layers}")

        hook_buffer = {}
        # 결과 누적용 딕셔너리
        accumulated = {layer: [] for layer in target_layers}
        handles = []

        def get_hook(name):
            # 클로저로 훅 함수 생성
            def hook(m, i, o): hook_buffer[name] = o

            return hook

        for name, module in self.model.named_modules():
            if name in target_layers:
                handles.append(module.register_forward_hook(get_hook(name)))

        self.model.eval()
        with torch.no_grad():
            # 미니배치를 전제로 하되, 풀배치여도 어차피 한 번 돌면 끝남
            for batch in tqdm(self.test_loader, desc="Extracting"):
                batch = batch.to(self.device)
                self.model(batch.x, batch.edge_index)  # Forward

                for layer in target_layers:
                    out = hook_buffer[layer]
                    # Slicing: 미니배치(Target Node만) vs 풀배치(Masking)
                    if self.test_mode == "mini":
                        out = out[:batch.batch_size]
                    else:
                        out = out[self.data.test_mask]

                    accumulated[layer].append(out.cpu())

        for h in handles: h.remove()  # 훅 제거

        # Concat
        return {k: torch.cat(v, dim=0) for k, v in accumulated.items()}

    # --- [Core Logic: Visualization] ---
    def _visualize_embeddings(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        UMAP을 사용하여 임베딩 시각화
        1. 필요한 라이브러리 임포트
        2. 샘플링 (최대 10,000개) - 더 많이 뽑으면 느려지고 메모리 터질 수 있음
        3. UMAP 변환 및 플롯 생성
        4. 로거를 통해 시각화 저장
        5. 예외 처리

        단, 속도가 매우 느려지니 필요 없을 때는 끌 것.

        - 추출 대상 레이어가 존재할 경우 그 레이어의 임베딩을 기준으로 시각화
        - 추출 대상 레이어가 없을 경우 output logits을 기준으로 시각화
        """
        try:
            import umap
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            log.warning("Skipping visualization (Missing umap-learn/matplotlib).")
            return

        log.info("[Visualization] Generating UMAP plot...")
        x = embeddings.detach().cpu().numpy()
        y = labels.detach().cpu().numpy().flatten()

        # 최대 샘플 수 1만개로 제한
        MAX_SAMPLES = 10000
        if x.shape[0] > MAX_SAMPLES:
            log.info(f"Subsampling {MAX_SAMPLES} from {x.shape[0]} points...")
            idx = np.random.choice(x.shape[0], MAX_SAMPLES, replace=False)
            x, y = x[idx], y[idx]

        try:
            # UMAP 변환
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
            embedding_2d = reducer.fit_transform(x)
            # 플롯 생성
            plt.figure(figsize=(10, 10))

            # 임베딩을 2D로 플롯
            scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
            plt.colorbar(scatter, ticks=range(len(np.unique(y))))
            plt.title(f"{self.cfg.dataset.name} - {self.cfg.model.name} UMAP")
            plt.axis('off')

            self.logger.save_visualization(plt)
        except Exception as e:
            log.error(f"Visualization failed: {e}")

    # --- [Main Flow: Final Test] ---
    def _run_final_test(self):
        """
        최종 테스트 및 시각화 진행
        1. 베스트 체크포인트 로드 (단, 없으면 기존에 로드한 모델 사용)
        2. 중간 레이어 Feature (Embedding) Extraction 수행
        3. 최종 테스트 평가 수행
        4. 결과 저장 및 시각화
        """
        log.info("--- Starting Final Test & Visualization ---")

        # 현재 디렉토리에 저장된 최고 성능 모델의 체크포인트 로드 시도
        best_path = os.path.join(self.logger.output_dir, "best_checkpoint.pth")
        try:
            self._load_checkpoint(self.model, best_path)
        except Exception:
            # 만약 그 디렉토리가 없을 경우 (주로 evaluate 세팅에서) 기존에 로드된 값이 사용되도록 건너뛰어짐
            pass

        target_layers = list(self.cfg.experiment.get("feature_extractor_layers", []))
        visualize = self.cfg.experiment.get("visualize", False)
        save_emb = self.cfg.experiment.get("save_embeddings", False)

        extracted_data = None  # 저장용 (Dict or Tensor)
        viz_tensor = None  # 시각화용 (Tensor)
        test_results = {}

        # 중간 embedding 추출을 요청하였을 경우
        if target_layers:
            # 중간 레이어 Feature Extraction 수행
            extracted_data = self._extract_features(target_layers)

            # 시각화용 텐서 선택 (마지막 레이어 기준)
            viz_tensor = extracted_data[target_layers[-1]]

            # 최종 테스트는 추출된 임베딩을 사용하여 평가
            test_results, _ = self.evaluator.evaluate(
                self.test_loader, self.test_mode,
                self.data.test_mask if self.test_mode == "full" else None,
                return_logits=False
            )

        # 중간 레이어를 별도로 지정하지 않았을 시
        else:
            # Evaluator 가 평가와 최종 logits 수집을 동시에 수행
            test_results, final_logits = self.evaluator.evaluate(
                self.test_loader, self.test_mode,
                self.data.test_mask if self.test_mode == "full" else None,
                return_logits=(visualize or save_emb)
            )
            extracted_data = final_logits
            viz_tensor = final_logits

        # 결과 저장
        log.info(f"Final Results: {test_results}")
        self.logger.save_final_results(test_results)

        # embeddings 값을 저장하라는 옵션이 존재할 경우 파일명 지정 후 저장
        if save_emb and extracted_data is not None:
            fname = "extracted_features.pt" if target_layers else "final_logits.pt"
            path = os.path.join(self.logger.output_dir, fname)
            torch.save(extracted_data, path)
            log.info(f"Saved embeddings to {path}")

        # 시각화
        if visualize and viz_tensor is not None:
            # Label도 Test Mask 씌워서 개수 맞춤
            labels_masked = self.data.y[self.data.test_mask]

            # 안전장치: 개수 다르면 최소 길이로 맞춤
            min_len = min(len(viz_tensor), len(labels_masked))
            self._visualize_embeddings(viz_tensor[:min_len], labels_masked[:min_len])

    def run(self):
        log.info(f"--- Experiment Start: {self.__class__.__name__} ---")
        # 제일 먼저 모델 복잡도 추출 및 저장
        self._profile_model_complexity()

        if self.cfg.experiment.get("mode", "train") == "train":
            # train 외의 다른 걸 지정하면 (test, evaluate 등등) 지정 모델에 대한 평가 하나만 진행
            # wandb 로깅 등도 모두 정상적으로 수행됨
            self._run_training()
        else:
            log.info("Skipping training (Inference Mode).")

        # 최종 테스트 및 시각화
        self._run_final_test()
        log.info(f"--- Experiment Finished ---")