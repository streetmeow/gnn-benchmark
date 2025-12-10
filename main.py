# main.py

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import logging
import datetime
import wandb

# --- '현장 감독' (Experiment) 클래스들을 임포트 ---
# (이 파일들은 experiment/ 디렉토리에 있다고 가정)
from scripts import BaseExperiment, Logger
from scripts.experiments import SimpleExperiment
from scripts.loggers import BaseLogger
import os

# from experiment.ensemble_experiment import EnsembleExperiment # (추후 앙상블 실험 추가 시)

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Global seed set to {seed}")


def run_experiment(cfg: DictConfig):
    """
    주요 환경만 세팅하고 나머지는 Experiment 클래스에 전부 위임

    1. 글로벌 환경 설정 (시드 고정 등)
    2. 로거 초기화
    3. 실험 전략 선택
    4. 실행 위임
    """
    logger = BaseLogger(cfg)
    logger.setup_local_file_logging()  # (로거 초기화)
    # --- 1. 글로벌 환경 설정 ---
    set_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- 2. 실험 전략 선택 ---
    experiment: BaseExperiment = None  # (타입 힌트)
    try:
        # 실험 전략에 따라 적절한 Experiment 지정해서 돌리기
        # config 의 experiment 섹션 name 지정
        if cfg.experiment.name == "simple":
            experiment = SimpleExperiment(cfg, logger)
        # 예시 실험 세팅
        # elif cfg.experiment.name == "distillation":
        #     experiment = DistillationExperiment(cfg)

        # experiment name 설정이 잘못 되어 있음
        else:
            raise ValueError(f"Unknown experiment name: {cfg.experiment.name}. Check 'configs/experiment/'")

        # --- 3. 실행 위임 ---
        # BaseExperiment.run()이 모든 것을 처리
        experiment.run()

        log.info("Main run finished.")
    except Exception as e:
        log.error(f"An error occurred during the experiment: {e}")
        # raise e
    finally:
        # logger 종료 작업
        logger.finish()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 한국 표준시(KST) 기준 폴더 제작을 위한 OmegaConf 리졸버 등록
    if not OmegaConf.has_resolver("kst"):
        OmegaConf.register_new_resolver("kst", lambda fmt: datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9))).strftime(fmt))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
