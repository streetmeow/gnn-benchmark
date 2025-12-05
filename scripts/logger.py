import torch
import torch.nn as nn
import os
import logging
import wandb
import datetime
import csv
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Optional, Dict, Any
import socket


log = logging.getLogger(__name__)


class Logger:
    """
    벤치마크의 모든 로깅을 담당하는 통합 로거.
    BaseExperiment에 의해 생성 및 소유되며,
    로컬 Hydra 출력 디렉토리와 WandB 서버에 동시 기록을 수행한다.

    [담당 작업]
    1. (Req 1) 소스 코드 아카이빙 (Config 기반)
    2. (Req 2) 실행/에러 로그 (main.py의 FileHandler가 처리, 이 클래스는 경로만 제공)
    3. (Req 3) 에폭별/최종 메트릭 (로컬 .csv/.yaml + WandB)
    4. (Req 4) 모델 체크포인트 (로컬 .pth + WandB Artifact)
    """

    def __init__(self, cfg: DictConfig):
        """
        Logger 초기화.
        Hydra 경로를 가져오고, WandB 세션을 시작하며, 코드 아카이빙을 수행.
        """
        self.cfg = cfg
        self.wandb_enabled = cfg.logging.use_wandb

        # Hydra가 생성한 최종 출력 디렉토리 경로를 가져옴
        try:
            self.output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            # Hydra가 아닌 환경(e.g., 일반 Python 실행)을 위한 폴백
            kst = datetime.timezone(datetime.timedelta(hours=9))
            self.output_dir = os.path.join("./output_logs", datetime.datetime.now(kst).strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(self.output_dir, exist_ok=True)
            log.warning(f"Hydra output_dir를 찾을 수 없습니다. 현재 디렉토리 사용: {self.output_dir}")

        # (Req 3) 에폭별 메트릭을 저장할 로컬 CSV 파일 경로
        self.epoch_csv_path = os.path.join(self.output_dir, "epoch_metrics.csv")
        self.epoch_csv_header_written = False

        if self.wandb_enabled:
            self._init_wandb()

        # (Req 1) 소스 코드 아카이빙 실행
        self.archive_source_code()

    def _init_wandb(self):
        """WandB 세션을 초기화 (API Key는 환경변수로 처리)."""
        cfg = self.cfg

        # 1. Config에서 '재료'를 긁어옴 (스크립트에서 조립)
        model_name = cfg.model.name.replace("/", "_")
        dataset_name = cfg.dataset.name.replace("/", "_")
        experiment_name = cfg.experiment.wandb_name
        seed = cfg.seed

        lr = cfg.train.lr
        layer_num = cfg.model.num_layers
        run_name = \
            f"{model_name}_{dataset_name}_lr{lr}_ly{layer_num}_wd{cfg.train.weight_decay}_dr{cfg.model.dropout}_hd{cfg.model.hidden_dim}"
        group_name = f"{experiment_name}_{model_name}"

        # 2. 'group' 이름 조립 (전략 + 모델 + 데이터셋)
        tags = [model_name, dataset_name, str(layer_num) + "layers", "v14"]
        if cfg.dataset.get("use_sampler", False):
            tags.append("sampler")
        if cfg.train.get("use_batchnorm", False):
            tags.append("batchnorm")

        # # 3. 'name' 이름 조립 (훈련 + 시드 + 타임스탬프)
        # kst = datetime.timezone(datetime.timedelta(hours=9))
        # timestamp = datetime.datetime.now(kst).strftime('%H%M%S')
        # run_name = f"{train_name}_seed_{seed}_{timestamp}"

        try:
            wandb.init(
                project=cfg.logging.project,
                group=group_name,
                name=run_name,
                tags=tags,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True  # (순차 실행 시 재초기화 허용)
            )
            wandb.config.update({
                "host_name": socket.gethostname(),  # 어느 서버에서 돌렸는지 (ex: lab-server-02)
                "output_dir": os.path.abspath(self.output_dir)  # 로컬 절대 경로
            }, allow_val_change=True)
            log.info(f"WandB initialized. Group: {experiment_name}, Run: {run_name}")
        except wandb.errors.AuthenticationError:
            log.error("WandB AuthenticationError. WANDB_API_KEY 환경 변수가 올바르게 설정되었는지 확인하세요.")
            self.wandb_enabled = False
        except Exception as e:
            log.error(f"WandB failed to initialize: {e}")
            self.wandb_enabled = False

    def setup_local_file_logging(self):
        """
        (Req 2) 로컬 'training.logging' 파일 핸들러를 설정합니다.
        main.py에서 이 메서드를 호출해야 합니다.
        """
        log_file_path = os.path.join(self.output_dir, "training.logging")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        )
        file_handler.setFormatter(formatter)

        has_console = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)

        if not has_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # 보고 싶은 레벨 설정
            console_handler.setFormatter(formatter)  # 파일이랑 똑같은 포맷 사용
            root_logger.addHandler(console_handler)

        # 루트 로거에 핸들러 추가
        logging.getLogger().addHandler(file_handler)
        log.info(f"Local file logging (Req 2) enabled at: {log_file_path}")

    def archive_source_code(self):
        """
        (Req 1) Config에 명시된 소스 코드를 아카이빙합니다.
        로컬: 'archived_source.txt' / WandB: 'Files' 탭
        """
        source_paths = self.cfg.logging.get("source_files", [])
        if not source_paths:
            log.warning("cfg.logging.source_files가 비어있어 코드 아카이빙을 건너뜁니다.")
            return

        archive_path = os.path.join(self.output_dir, "archived_source.txt")

        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(f"--- Archived Source Code for Run: {self.output_dir} ---\n")

            for path in source_paths:
                try:
                    with open(path, "r", encoding="utf-8") as src_file:
                        f.write(f"\n{'=' * 20} FILE: {path} {'=' * 20}\n")
                        f.write(src_file.read())

                    if self.wandb_enabled:
                        wandb.save(path, base_path=".")
                except FileNotFoundError:
                    log.warning(f"Source file not found (skipped): {path}")
                except Exception as e:
                    log.error(f"Error archiving {path}: {e}")

        log.info(f"Archived {len(source_paths)} source files to {archive_path}")

    def log_epoch_metrics(self, metrics_dict: Dict[str, Any], step: int):
        """
        (Req 3 - Epoch) 에폭별 메트릭을 로컬 CSV와 WandB에 기록합니다.
        """
        # 1. 로컬 CSV 기록
        try:
            # [수정 1] 키 목록을 미리 가져옵니다.
            current_keys = list(metrics_dict.keys())

            # 헤더 작성 (첫 에폭에만 실행)
            if not self.epoch_csv_header_written:
                header = ["epoch"] + current_keys
                with open(self.epoch_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                self.epoch_csv_header_written = True

            # [수정 2] 'header' 변수 대신 'current_keys'를 사용하여 row 생성
            # (metrics_dict.get(k, "")는 혹시 모를 키 누락 방지용)
            row = [step] + [metrics_dict.get(k, "") for k in current_keys]

            with open(self.epoch_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            log.warning(f"Failed to write epoch metrics to CSV: {e}")

        # 2. WandB 실시간 전송
        if self.wandb_enabled:
            wandb.log(metrics_dict, step=step)

    def save_final_results(self, final_metrics: Dict[str, Any]):
        """
        (Req 3 - Final) 최종 성능을 로컬 YAML과 WandB Summary에 기록합니다.
        """
        # 1. 로컬 YAML 저장
        local_path = os.path.join(self.output_dir, "final_results.yaml")
        try:
            OmegaConf.save(OmegaConf.create(final_metrics), local_path)
            log.info(f"Final results saved to {local_path}")
        except Exception as e:
            log.error(f"Failed to save final results to YAML: {e}")

        # 2. WandB Summary 탭에 기록
        if self.wandb_enabled:
            wandb.summary.update(final_metrics)

    def save_model_complexity(self, complexity: Dict[str, Any]):
        log_file_path = os.path.join(self.output_dir, "model_complexity.txt")
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                for key, value in complexity.items():
                    f.write(f"{key}: {value}\n")
            log.info(f"Model complexity saved to {log_file_path}")
        except Exception as e:
            log.error(f"Failed to save model complexity: {e}")
        if self.wandb_enabled:
            wandb.summary.update(complexity)

    def save_checkpoint(self, model: nn.Module, filename: str = "best_model.pth"):
        """
        (Req 4) 모델 가중치를 로컬 .pth 파일과 WandB Artifacts로 저장합니다.
        """
        local_path = os.path.join(self.output_dir, filename)
        try:
            torch.save(model.state_dict(), local_path)
            log.info(f"Model checkpoint saved to {local_path}")
        except Exception as e:
            log.error(f"Failed to save model checkpoint: {e}")

        # WandB Artifacts로 모델 저장 (버전 관리)
        if self.wandb_enabled:
            try:
                abs_path = os.path.abspath(local_path)
                uri = f"file://{abs_path}"

                artifact_name = f"{self.cfg.model.name}-{self.cfg.dataset.name}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type='model',
                    description=f"Reference to model {wandb.run.name}"
                )
                artifact.add_reference(uri, name=filename, checksum=False)

                wandb.log_artifact(artifact)
                log.info(f"Logged model reference to WandB: {uri}")

                # (선택 사항) WandB Config에 로컬 경로 텍스트도 남기기
                wandb.config.update({"hydra_output_dir": os.path.abspath(self.output_dir)}, allow_val_change=True)

            except Exception as e:
                log.error(f"Failed to log model artifact: {e}")

    def save_visualization(self, plt_module: Any, filename: str = "embedding_visualization.png"):
        local_path = os.path.join(self.output_dir, filename)
        try:
            plt_module.savefig(local_path, dpi=300, bbox_inches='tight')
            plt_module.close()
            log.info(f"Visualization saved to {local_path}")
        except Exception as e:
            log.error(f"Failed to save visualization: {e}")
            return
        if self.wandb_enabled:
            try:
                wandb.log({"visualization/umap": wandb.Image(local_path)})
                log.info(f"Logged visualization to WandB: {filename}")
            except Exception as e:
                log.error(f"Failed to log visualization to WandB: {e}")

    def finish(self):
        """실험 종료 시 WandB 세션을 닫습니다."""
        if self.wandb_enabled and wandb.run:
            wandb.finish()
            log.info("WandB run finished.")