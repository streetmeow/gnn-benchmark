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

# ============================================================
# 🔥 2) 공통: epoch 설정
# ============================================================
EPOCH_TABLE = {
    "ogbn-products": 100,
    "ogbn-arxiv": 80,
    "pubmed": 140,
    "actor": 250,
    "citeseer": 180,
    "cora": 150,
}


# ============================================================
# 🔥 3) dataset/model 조합 단위로만 grid search 수행
# ============================================================
def run_grid_search(target_datasets=None, target_models=None):
    if not OmegaConf.has_resolver("kst"):
        OmegaConf.register_new_resolver(
            "kst",
            lambda fmt: datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=9))
            ).strftime(fmt)
        )

    # 전체 대상 지정
    all_pairs = list(HP_SEARCH_SPACE.keys())

    # 만약 특정 dataset/model만 선택했다면 filtering
    if target_datasets is not None:
        all_pairs = [p for p in all_pairs if p[0] in target_datasets]
    if target_models is not None:
        all_pairs = [p for p in all_pairs if p[1] in target_models]

    print(f"🎯 Target pairs: {all_pairs}")

    with initialize(version_base=None, config_path="configs"):

        for (dataset, model) in all_pairs:
            if dataset == "ogbn-products":
                continue  # 시간 관계상 ogbn-products는 제외

            hp = HP_SEARCH_SPACE[(dataset, model)]

            # Cartesian product for this dataset/model
            grid_keys = ["hidden", "lr", "dropout", "wd"]
            grid_values = [hp[k] for k in grid_keys]
            grid = list(itertools.product(*grid_values))

            print(f"\n===============================================")
            print(f"🚀 Running {dataset} / {model}")
            print(f"🔧 Total configs: {len(grid)}")
            print("===============================================\n")

            # epoch 설정
            epochs = EPOCH_TABLE[dataset]

            # layer 개수대로 반복 (예: [2], [2,3])
            for layer in hp["layer"]:
                for i, combo in enumerate(grid):
                    hidden, lr, dropout, wd = combo

                    print(
                        f"[{dataset}/{model}] layer={layer} "
                        f"({i+1}/{len(grid)}) | "
                        f"hd={hidden}, lr={lr}, dr={dropout}, wd={wd}"
                    )

                    overrides = [
                        f"dataset={dataset}",
                        f"model={model}",
                        f"gpu_id=0",
                        "seed=5",

                        # HP 적용
                        f"model.num_layers={layer}",
                        f"model.hidden_dim={hidden}",
                        f"model.dropout={dropout}",
                        f"train.lr={lr}",
                        f"train.weight_decay={wd}",

                        # epochs
                        f"train.epochs={epochs}",
                    ]

                    # sampler 여부
                    is_large = dataset in ["ogbn-products", "ogbn-arxiv"]
                    if is_large or model == "graphsage":
                        overrides.append("dataset.use_sampler=true")
                        bs = 1024 if is_large else 512
                        overrides.append(f"sampler.batch_size={bs}")
                    else:
                        overrides.append("dataset.use_sampler=false")

                    # layer-specific model sizes
                    if layer == 2:
                        overrides.append("model.sizes=[15,10]")
                    else:
                        overrides.append("model.sizes=[15,10,5]")

                    try:
                        if i == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        cfg = compose(config_name="config", overrides=overrides)
                        run_experiment(cfg)

                    except Exception as e:
                        print(f"❌ Error in {dataset}/{model} @layer={layer}: {e}")
                        traceback.print_exc()

                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gc.collect()
    torch.cuda.empty_cache()

    # 필요하면 일부만 선택 가능:
    # run_grid_search(target_datasets=["cora"], target_models=["gcn"])

    run_grid_search()
