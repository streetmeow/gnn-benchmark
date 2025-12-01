# multi_run.py (Grid Search 완성본)
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
import datetime
import torch
import gc
import traceback
import itertools
import os

# [핵심] main.py에서 알맹이 함수 가져오기
from main import run_experiment


HP_SEARCH_SPACE = {

    # ======================

    # 1) CITESEER

    # ======================

    ("citeseer", "gcn"): {

        "layer": [2],                 # 2-layer 더 안정적

        "hidden": [64, 128],

        "lr": [0.01, 0.005],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },

    ("citeseer", "gat"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001, 0.005, 0.01],

        "dropout": [0.2, 0.5],

        "wd": [5e-4],

    },

    ("citeseer", "gin"): {

        "layer": [2],

        "hidden": [64, 128],

        "lr": [0.001, 0.0005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("citeseer", "graphsage"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001, 0.005, 0.01],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },



    # ======================

    # 2) CORA

    # ======================

    ("cora", "gcn"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("cora", "gat"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.01, 0.001, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("cora", "gin"): {

        "layer": [2],

        "hidden": [128],

        "lr": [0.001, 0.0005, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("cora", "graphsage"): {

        "layer": [3],

        "hidden": [64, 128],

        "lr": [0.001, 0.01, 0.005],

        "dropout": [0.5],

        "wd": [5e-5, 5e-4],

    },



    # ======================

    # 3) PUBMED

    # ======================

    ("pubmed", "gcn"): {

        "layer": [2, 3],

        "hidden": [64, 128],

        "lr": [0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-4],

    },

    ("pubmed", "gat"): {

        "layer": [2, 3],

        "hidden": [64, 128],

        "lr": [0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-4],

    },

    ("pubmed", "gin"): {

        "layer": [2, 3],

        "hidden": [64, 128],

        "lr": [0.005, 0.01],

        "dropout": [0.2, 0.5],

        "wd": [5e-4],

    },

    ("pubmed", "graphsage"): {

        "layer": [2, 3],

        "hidden": [64, 128],

        "lr": [0.001],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },



    # ======================

    # 4) ACTOR

    # ======================

    ("actor", "gcn"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001, 0.005],

        "dropout": [0.5, 0.2],  # dropout 영향 명확히 긍정적

        "wd": [5e-5, 5e-4],

    },

    ("actor", "gat"): {

        "layer": [2],

        "hidden": [64, 128],

        "lr": [0.005, 0.01],

        "dropout": [0.5, 0.2],

        "wd": [5e-4],

    },

    ("actor", "gin"): {

        "layer": [2],

        "hidden": [64, 128],

        "lr": [0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("actor", "graphsage"): {

        "layer": [3],   # actor에서 GraphSAGE는 3 layer 우세

        "hidden": [64, 128],

        "lr": [0.001, 0.005, 0.01],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },



    # ======================

    # 5) OGBN-PRODUCTS

    # ======================

    ("ogbn-products", "gcn"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001, 0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "gat"): {

        "layer": [3],

        "hidden": [128, 64],

        "lr": [0.001, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },

    ("ogbn-products", "gin"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("ogbn-products", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001, 0.005, 0.01],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },



    # ======================

    # 6) OGBN-ARXIV

    # ======================

    ("ogbn-arxiv", "gcn"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001, 0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },

    ("ogbn-arxiv", "gat"): {

        "layer": [3],

        "hidden": [64, 128],

        "lr": [0.001, 0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5],

    },

    ("ogbn-arxiv", "gin"): {

        "layer": [2, 3],

        "hidden": [128],

        "lr": [0.001],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

    ("ogbn-arxiv", "graphsage"): {

        "layer": [3],

        "hidden": [128],

        "lr": [0.001, 0.01, 0.005],

        "dropout": [0.2, 0.5],

        "wd": [5e-5, 5e-4],

    },

}


def run_grid_search():
    # KST 리졸버 등록
    if not OmegaConf.has_resolver("kst"):
        OmegaConf.register_new_resolver("kst", lambda fmt: datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9))).strftime(fmt))

    # 1. 탐색 공간 정의
    search_space = {
        # "dataset": ["cora", "citeseer", "pubmed", "ogbn-arxiv", "ogbn-products", "actor"],
        "dataset": ["ogbn-arxiv"],
        "model": ["gcn", "graphsage", "gat", "gin"],
        "seed": [5],
        # 하이퍼파라미터
        "model.hidden_dim": [64, 128],  # 예시로 줄임
        "model.dropout": [0.2, 0.5],
        "train.lr": [0.001, 0.01],
        "train.weight_decay": [5e-4, 5e-5]
    }

    # 2. Cartesian Product 생성
    keys, values = zip(*search_space.items())
    combinations = list(itertools.product(*values))

    print(f"🚀 Total Configurations to run: {len(combinations)}")

    with initialize(version_base=None, config_path="configs"):

        for i, combination in enumerate(combinations):
            # 현재 파라미터 딕셔너리 생성
            param_dict = dict(zip(keys, combination))

            # 진행상황 출력
            print(f"\n==========================================================")
            print(f"🧩 [Grid Search layer 2 {i + 1}/{len(combinations)}] Params: {param_dict}")

            d_name = param_dict["dataset"]
            m_name = param_dict["model"]

            # 3. Overrides 리스트 생성 (기본 파라미터)
            overrides = []
            for k, v in param_dict.items():
                overrides.append(f"{k}={v}")
            overrides.append("gpu_id=0")

            # [WandB Grouping] 한 눈에 보기 좋게 그룹 이름 설정
            # overrides.append(f"logging.experiment_strategy_name=GridSearch_v1")

            # 4. 조건부 로직 (Logic) - 안전장치 포함
            is_large_dataset = d_name in ["ogbn-products", "ogbn-arxiv"]

            # (1) 샘플러 활성화 여부 및 배치 사이즈 결정
            if is_large_dataset or m_name == "graphsage":
                overrides.append("dataset.use_sampler=true")
                overrides.append("sampler.batch_size=512")

                # 데이터셋별 배치 사이즈
                if is_large_dataset:
                    overrides.append("sampler.batch_size=1024")
            else:
                overrides.append("dataset.use_sampler=false")

            if d_name == "ogbn-products":
                overrides.append("train.epochs=100")
            elif d_name == "ogbn-arxiv":
                overrides.append("train.epochs=80")
            elif d_name == "pubmed":
                overrides.append("train.epochs=140")
            elif d_name == "actor":
                overrides.append("train.epochs=250")
            elif d_name == "citeseer":
                overrides.append("train.epochs=180")
            elif d_name == "cora":
                overrides.append("train.epochs=150")

            # (2) 모델별 레이어 및 샘플러 사이즈 매칭 (중요!)
            overrides.append("model.num_layers=2")
            overrides.append("model.sizes=[15,10]")

            try:
                if i == 0:
                    gc.collect()  # 1. 파이썬 쓰레기 수거 (참조 잃은 객체 삭제)
                    torch.cuda.empty_cache()  # 2. PyTorch가 잡고 있는 빈 메모리 캐시 해제
                    torch.cuda.synchronize()
                # 5. Config 조립 및 실행
                cfg = compose(config_name="config", overrides=overrides)
                run_experiment(cfg)

            except Exception as e:
                print(f"❌ Error in experiment layer 2 {i + 1}: {e}")
                traceback.print_exc()

            finally:
                # 6. 메모리 청소
                gc.collect()
                torch.cuda.empty_cache()

        for i, combination in enumerate(combinations):
            # 현재 파라미터 딕셔너리 생성
            param_dict = dict(zip(keys, combination))

            # 진행상황 출력
            print(f"\n==========================================================")
            print(f"🧩 [Grid Search layer 3 {i + 1}/{len(combinations)}] Params: {param_dict}")

            d_name = param_dict["dataset"]
            m_name = param_dict["model"]

            # 3. Overrides 리스트 생성 (기본 파라미터)
            overrides = []
            for k, v in param_dict.items():
                overrides.append(f"{k}={v}")
            overrides.append("gpu_id=0")

            # [WandB Grouping] 한 눈에 보기 좋게 그룹 이름 설정
            # overrides.append(f"logging.experiment_strategy_name=GridSearch_v1")

            # 4. 조건부 로직 (Logic) - 안전장치 포함
            is_large_dataset = d_name in ["ogbn-products", "ogbn-arxiv"]

            # (1) 샘플러 활성화 여부 및 배치 사이즈 결정
            if is_large_dataset or m_name == "graphsage":
                overrides.append("dataset.use_sampler=true")
                overrides.append("sampler.batch_size=512")

                # 데이터셋별 배치 사이즈
                if is_large_dataset:
                    overrides.append("sampler.batch_size=1024")
            else:
                overrides.append("dataset.use_sampler=false")

            if d_name == "ogbn-products":
                overrides.append("train.epochs=100")
            elif d_name == "ogbn-arxiv":
                overrides.append("train.epochs=80")
            elif d_name == "pubmed":
                overrides.append("train.epochs=140")
            elif d_name == "actor":
                overrides.append("train.epochs=250")
            elif d_name == "citeseer":
                overrides.append("train.epochs=180")
            elif d_name == "cora":
                overrides.append("train.epochs=150")

            # (2) 모델별 레이어 및 샘플러 사이즈 매칭 (중요!)
            overrides.append("model.num_layers=3")
            overrides.append("model.sizes=[15,10,5]")

            try:
                # 5. Config 조립 및 실행
                cfg = compose(config_name="config", overrides=overrides)
                run_experiment(cfg)

            except Exception as e:
                print(f"❌ Error in experiment layer 3 - {i + 1}: {e}")
                traceback.print_exc()

            finally:
                # 6. 메모리 청소
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    run_grid_search()
