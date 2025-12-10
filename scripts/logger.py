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
import subprocess
import psutil
import threading
import time


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
        self.output_dir = self.setup_output_dir()

        # (Req 3) 에폭별 메트릭을 저장할 로컬 CSV 파일 경로
        self.epoch_csv_path = os.path.join(self.output_dir, "epoch_metrics.csv")
        self.epoch_csv_header_written = False
        self.tags = []
        self._monitor_running = False
        self._monitor_thread = None
        self.system_stats = {
            "gpu_util": [],
            "gpu_mem": [],
            "cpu": [],
            "ram": []
        }

        if self.wandb_enabled:
            self._init_wandb()

        # (Req 1) 소스 코드 아카이빙 실행
        self.archive_source_code()

    def setup_output_dir(self) -> str:
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            kst = datetime.timezone(datetime.timedelta(hours=9))
            output_dir = os.path.join("./output_logs", datetime.datetime.now(kst).strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(output_dir, exist_ok=True)
            log.warning(f"Hydra output_dir를 찾을 수 없습니다. 현재 디렉토리 사용: {output_dir}")
            return output_dir

    def configure_wandb(self):
        cfg = self.cfg

        # 1. Config에서 '재료'를 긁어옴 (스크립트에서 조립)
        model_name = cfg.model.name.replace("/", "_")
        dataset_name = cfg.dataset.name.replace("/", "_")
        experiment_name = cfg.experiment.wandb_name
        seed = cfg.seed
        lr = cfg.train.lr
        layer_num = cfg.model.num_layers
        run_name = (
            f"{model_name}_{dataset_name}_lr{lr}"
            f"_ly{layer_num}_wd{cfg.train.weight_decay}"
            f"_dr{cfg.model.dropout}_hd{cfg.model.hidden_dim}"
        )
        group_name = f"{experiment_name}_{model_name}"

        tags = [model_name, dataset_name, str(layer_num) + "layers", "seed" + str(seed)]
        if cfg.dataset.get("use_sampler", False):
            tags.append("sampler")
        if cfg.train.get("use_batchnorm", False):
            tags.append("batchnorm")
        self.tags = tags
        self.extend_tags()
        return dict(
            project=cfg.logging.project,
            group=group_name,
            name=run_name,
            tags=self.tags
        )

    def extend_tags(self):
        pass

    # ================================
    #  System Monitoring (GPU/CPU/RAM)
    # ================================

    def _get_active_gpu_usage(self):
        """
        현재 Python 프로세스가 실제로 사용 중인 GPU의 utilization/memory를 반환한다.
        CUDA_VISIBLE_DEVICES를 반영해 system GPU ID로 매핑함.
        """
        try:
            # 현재 파이썬에서 활성화된 local GPU ID (보통 0)
            local_id = torch.cuda.current_device()

            # CUDA_VISIBLE_DEVICES 설정 확인
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible:
                mapping = [int(x) for x in visible.split(",")]
                system_gpu_id = mapping[local_id]
            else:
                system_gpu_id = local_id

            # 해당 system_gpu_id만 nvidia-smi로 쿼리
            cmd = (
                f"nvidia-smi --id={system_gpu_id} "
                f"--query-gpu=utilization.gpu,memory.used "
                f"--format=csv,noheader,nounits"
            )
            out = subprocess.check_output(cmd, shell=True).decode().strip()

            # 공백 제거 후 split
            parts = [p.strip() for p in out.replace("MiB", "").replace("%", "").split(",")]

            if len(parts) >= 2:
                util = int(parts[0])
                mem = int(parts[1])
                return util, mem
            else:
                return None, None

        except Exception:
            return None, None

    def start_system_monitor(self, interval: float = 5.0):
        """
        별도 thread로 GPU/CPU/RAM usage를 주기적으로 wandb로 기록.
        평균/최대값 계산을 위해 내부 리스트에도 누적 저장.
        """
        if not self.wandb_enabled:
            return

        self._monitor_running = True

        def monitor_loop():
            while self._monitor_running:
                metrics = {}

                # GPU usage (utilization %, memory MB)
                util, mem = self._get_active_gpu_usage()
                if util is not None:
                    metrics["system/gpu_util"] = util
                    metrics["system/gpu_mem_mb"] = mem
                    self.system_stats["gpu_util"].append(util)
                    self.system_stats["gpu_mem"].append(mem)

                # CPU usage
                try:
                    cpu_val = psutil.cpu_percent(interval=0.1)
                    metrics["system/cpu"] = cpu_val
                    self.system_stats["cpu"].append(cpu_val)
                except Exception:
                    pass

                # RAM usage
                try:
                    ram_val = psutil.virtual_memory().percent
                    metrics["system/ram"] = ram_val
                    self.system_stats["ram"].append(ram_val)
                except Exception:
                    pass

                if metrics:
                    wandb.log(metrics)

                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_system_monitor(self):
        """Monitoring thread 종료"""
        if hasattr(self, "_monitor_running"):
            self._monitor_running = False

    def _init_wandb(self):
        """WandB 세션을 초기화 (API Key는 환경변수로 처리)."""
        cfg = self.cfg

        # # 3. 'name' 이름 조립 (훈련 + 시드 + 타임스탬프)
        # kst = datetime.timezone(datetime.timedelta(hours=9))
        # timestamp = datetime.datetime.now(kst).strftime('%H%M%S')
        # run_name = f"{train_name}_seed_{seed}_{timestamp}"

        try:
            wandb_args = self.configure_wandb()
            wandb.init(
                **wandb_args,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True  # (순차 실행 시 재초기화 허용)
            )
            wandb.config.update({
                "host_name": socket.gethostname(),  # 어느 서버에서 돌렸는지 (ex: lab-server-02)
                "output_dir": os.path.abspath(self.output_dir)  # 로컬 절대 경로
            }, allow_val_change=True)
            log.info(f"WandB initialized. Args: {wandb_args}")
            self.start_system_monitor(interval=5.0)
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
        if self.wandb_enabled:

            # 모니터링 종료
            self.stop_system_monitor()

            # 평균/최대 계산 후 summary 업데이트
            try:
                summary_stats = {}
                for key, arr in self.system_stats.items():
                    if len(arr) > 0:
                        summary_stats[f"system/{key}_avg"] = sum(arr) / len(arr)
                        summary_stats[f"system/{key}_max"] = max(arr)

                wandb.summary.update(summary_stats)
                log.info(f"System summary stats saved: {summary_stats}")

            except Exception as e:
                log.error(f"Failed to compute system summary stats: {e}")

            # wandb 종료
            if wandb.run:
                wandb.finish()
                log.info("WandB run finished.")