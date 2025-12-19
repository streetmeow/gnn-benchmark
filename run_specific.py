# multi_run_cpf.py
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
import datetime
import torch
import gc
import traceback
import os

from main import run_experiment
from run_settings.cfp_teacher_table import CPF_TEACHER_TABLE


# ============================================================
# CPF Student (논문 재현: 고정)
# ============================================================
CPF_STUDENT_CONFIG = {
    "hidden_dim": 128,
    "dropout": 0.5,
    "plp_steps": 10,
    "lr": 1e-3,
    "weight_decay": 0.0,
}

EPOCH_TABLE = {
    "cora": 100,
    "citeseer": 100,
    "pubmed": 100,
    "actor": 100,
    "texas": 100,
    "cornell": 100,
}

PATIENCE_TABLE = {
    "cora": 30,
    "citeseer": 30,
    "pubmed": 30,
    "actor": 30,
    "texas": 30,
    "cornell": 30,
}


# ============================================================
# CPF Runner
# ============================================================
def run_cpf_experiments(target_datasets=None, target_teachers=None):
    if not OmegaConf.has_resolver("kst"):
        OmegaConf.register_new_resolver(
            "kst",
            lambda fmt: datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=9))
            ).strftime(fmt)
        )

    all_pairs = list(CPF_TEACHER_TABLE.keys())

    if target_datasets is not None:
        all_pairs = [p for p in all_pairs if p[0] in target_datasets]
    if target_teachers is not None:
        all_pairs = [p for p in all_pairs if p[1] in target_teachers]

    print(f"🎯 CPF target pairs: {all_pairs}")

    with initialize(version_base=None, config_path="configs"):
        for gam in [1.0, 3.0]:
            for dataset, teacher_model in all_pairs:
                for seed_num in range(1, 6):
                    for lambda_gate in [0.0, 0.1, 0.3, 1.0]:
                        teacher_cfg = CPF_TEACHER_TABLE[(dataset, teacher_model)]

                        epochs = EPOCH_TABLE[dataset]
                        patience = PATIENCE_TABLE[dataset]

                        print("\n===============================================")
                        print(f"🚀 CPF | dataset={dataset} | teacher={teacher_model}")
                        print(f"📦 checkpoint={teacher_cfg['checkpoint']} | seed={seed_num}")
                        print("===============================================\n")

                        overrides = [
                            # --------------------------------------------------
                            # experiment
                            # --------------------------------------------------
                            # "experiment=cpf",

                            # --------------------------------------------------
                            # dataset (CPF는 full-batch 가정)
                            # --------------------------------------------------
                            f"dataset={dataset}",
                            "dataset.use_sampler=false",

                            # --------------------------------------------------
                            # teacher (여기가 핵심 수정 포인트)
                            # --------------------------------------------------
                            f"teacher={teacher_cfg['model']}",                  # config group
                            f"teacher_checkpoint={teacher_cfg['checkpoint']}",
                            f"teacher.num_layers={teacher_cfg['num_layers']}",
                            f"teacher.hidden_dim={teacher_cfg['hidden_dim']}",
                            f"teacher.dropout={teacher_cfg['dropout']}",
                            # f"teacher.lr={teacher_cfg['lr']}",
                            # f"teacher.weight_decay={teacher_cfg['weight_decay']}",

                            # --------------------------------------------------
                            # student (CPF, 고정)
                            # --------------------------------------------------
                            f"model.hidden_dim={CPF_STUDENT_CONFIG['hidden_dim']}",
                            f"model.dropout={CPF_STUDENT_CONFIG['dropout']}",
                            # f"+model.plp_steps={CPF_STUDENT_CONFIG['plp_steps']}",

                            # --------------------------------------------------
                            # training (student only)
                            # --------------------------------------------------
                            f"train.lr={CPF_STUDENT_CONFIG['lr']}",
                            f"train.weight_decay={CPF_STUDENT_CONFIG['weight_decay']}",
                            f"train.epochs={epochs}",
                            f"train.patience={patience}",
                            f"train.lambda_gate={lambda_gate}",
                            f"train.gamma={gam}",
                            "train.use_early_stopping=true",

                            # --------------------------------------------------
                            # system
                            # --------------------------------------------------
                            f"seed={seed_num}",
                            "gpu_id=0",
                        ]

                        try:
                            gc.collect()
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                            cfg = compose(config_name="config", overrides=overrides)
                            run_experiment(cfg)

                        except Exception as e:
                            print(f"❌ CPF Error @ {dataset}/{teacher_model}: {e}")
                            traceback.print_exc()

                        finally:
                            gc.collect()
                            torch.cuda.empty_cache()



# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    gc.collect()
    torch.cuda.empty_cache()

    run_cpf_experiments()
