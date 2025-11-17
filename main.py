# main.py

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import logging

# --- '현장 감독' (Experiment) 클래스들을 임포트 ---
# (이 파일들은 experiment/ 디렉토리에 있다고 가정)
from scripts import BaseExperiment
from scripts.experiments.simple_experiment import SimpleExperiment

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    [하이레벨 지휘자]
    Hydra로부터 cfg를 받아 '환경'을 설정하고,
    '어떤 실험(Experiment)'을 실행할지 '선택'한 뒤,
    '실행'을 위임한다.
    """

    # --- 1. 글로벌 환경 설정 ---
    set_seed(cfg.seed)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # (추후 WandB 로거 설정)
    # wandb.init(project="gnn-distill-benchmark", config=OmegaConf.to_container(cfg))

    # --- 2. '실험(현장 감독)' 전략 선택 ---
    experiment: BaseExperiment = None  # (타입 힌트)

    # 'cfg.experiment.name' (e.g., "simple", "distillation")을 읽음
    if cfg.experiment.name == "simple":
        experiment = SimpleExperiment(cfg)
    #
    # elif cfg.experiment.name == "distillation":
    #     experiment = DistillationExperiment(cfg)

    # (추후 앙상블 실험 추가 시)
    # elif cfg.experiment.name == "ensemble_2teacher":
    #     experiment = EnsembleExperiment(cfg)

    else:
        raise ValueError(f"Unknown experiment name: {cfg.experiment.name}. Check 'configs/experiment/'")

    # --- 3. 실행 위임 ---
    # (BaseExperiment.run()이 모든 것을 처리)
    experiment.run()

    log.info("Main run finished.")
    # (WandB 종료)
    # wandb.finish()


if __name__ == "__main__":
    main()
